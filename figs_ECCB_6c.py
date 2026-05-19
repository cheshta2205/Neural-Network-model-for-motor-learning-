"""
can we refigs_6c.py

Single self-contained script that regenerates all paper figures for the 6-context
EC→CB model from the saved checkpoint. No retraining — just loads weights,
regenerates the val rollouts, and writes every figure into Paper/figs/.

Figures produced (all in M.FIG_DIR):
    01 training.png                    — train/val MSE curves
    02 pca.png                         — PCA of z_t coloured by context
    03 confusion.png                   — context confusion matrix
    04 perdim.png                      — univariate per-neuron decode accuracy
    05 decodetime.png                  — joint decode accuracy over timestep
    06 decoder_importance.png          — joint-decoder weight norm per neuron
    07 svd_singular_values.png         — context-subspace singular values (5 nonzero)
    08 svd_U_heatmap.png               — U: per-context loadings on each SVD axis
    09 svd_context_axes.png            — Vᵀ: per-neuron contribution to each axis
    10 cb_context_alignment.png        — does CB read EC's context subspace?
    11 ablation_per_neuron.png         — ΔMSE on zeroing each z neuron
    12 ablation_progressive.png        — cumulative ablation sorted by decode acc
    13 ctx_subspace_ablation.png       — projecting z off the top-k context axes
    14 param_encoding_all.png          — per-neuron R² for θ, γ, g (univariate)
    15 param_one_glm.png               — joint ridge |coef| per neuron for θ/γ/g
    16 complement_subspace_ablation.png — what CB loses without the non-context dims
    17 z_variable_decoding.png         — joint R² for state, action, velocity, etc.

    EC→CB synapse readout analyses (purely synapse + projection + visualization):
    18 synapse_singular_values.png      — W singular values (32 bars)
    19 synapse_top3_axes_context.png    — top-3 CB-preferred axes (4-view 3D), coloured by context
    20 synapse_top3_axes_state.png      — top-3 CB-preferred axes (4-view 3D), coloured by state x
    21 synapse_top3_axes_velocity.png   — top-3 CB-preferred axes (4-view 3D), coloured by |V_t|
    22 synapse_zproj_ablation.png       — ablate top-k CB-preferred axes from z @ U_c → CB MSE

    EC readout analyses (h_ec rotated into the readout's preferred basis):
    23 ec_readout_singular_values.png   — readout W singular values (32 bars)
    24 ec_readout_top3_axes_context.png — top-3 readout-preferred axes (4-view 3D), context
    25 ec_readout_top3_axes_state.png   — same, coloured by state x
    26 ec_readout_top3_axes_state_xy.png— same, coloured by (x, y) via RGB
    27 ec_readout_top3_axes_velocity.png— same, coloured by |V_t|

    CB layer-1 / layer-2 PCA analyses (no weight matrix, just data PCA):
    28 cb_layer1_top3_axes_*.png        — top-3 PCs of CB layer-1 (4 colorings)
    29 cb_layer2_top3_axes_*.png        — top-3 PCs of CB layer-2 (4 colorings)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Pull all shared definitions (models, contexts, data gen) from the training script
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "train_mod", os.path.join(HERE, "ECtoCB_train_MSE_6c.py")
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

TAG = "EC→CB | velocity dynamics | 6 contexts (no scale)"
COLORS = {
    0: "steelblue", 1: "cornflowerblue",
    2: "seagreen",  3: "limegreen",
    4: "purple",    5: "violet",
}


# ── small util ─────────────────────────────────────────────────────────────────
def fig_path(name):
    return os.path.join(M.FIG_DIR, name)


def collect_z_and_context(ec, cb, va_trans, va_ctx, device):
    """Run val rollouts through EC+CB, collect z and context labels."""
    all_z, all_ctype = [], []
    with torch.no_grad():
        for i in range(0, len(va_trans), M.BATCH_SIZE):
            trans = va_trans[i:i+M.BATCH_SIZE].to(device)
            ctx   = va_ctx[i:i+M.BATCH_SIZE]
            z, _  = M.run_sequence(ec, cb, trans, device)
            B, T, D = z.shape
            all_z.append(z.cpu().reshape(B * T, D))
            all_ctype.append(ctx[:, :, 2].reshape(B * T))
    all_z     = torch.cat(all_z).numpy()
    all_ctype = torch.cat(all_ctype).numpy().astype(int)
    return all_z, all_ctype


def make_cb_mse_fn(cb, va_trans, device):
    """Closure returning CB MSE when CB is fed a given z_seq (N, T, LATENT_DIM)."""
    N_val = len(va_trans)
    def cb_mse_with_z(z_seq_np):
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, N_val, M.BATCH_SIZE):
                trans    = va_trans[i:i+M.BATCH_SIZE].to(device)
                z        = torch.tensor(z_seq_np[i:i+M.BATCH_SIZE], dtype=torch.float32, device=device)
                s_curr   = trans[:, :, :2].reshape(-1, 2)
                a_curr   = trans[:, :, 2:4].reshape(-1, 2)
                s_target = trans[:, :, 4:6].reshape(-1, 2)
                z_flat   = z.reshape(-1, M.LATENT_DIM)
                total   += F.mse_loss(cb(s_curr, a_curr, z_flat), s_target, reduction='sum').item()
                count   += s_target.shape[0]
        return total / count
    return cb_mse_with_z


# ── fig generators ─────────────────────────────────────────────────────────────
def fig_training(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_mse"], label="train MSE", color="steelblue")
    ax.plot(history["val_mse"],   label="val MSE",   color="tomato")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title(f"Training curve — {TAG}")
    ax.legend(); plt.tight_layout()
    out = fig_path("training.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_pca(all_z, all_ctype):
    n_plot = min(2000, len(all_z))
    idx    = np.random.default_rng(0).choice(len(all_z), n_plot, replace=False)
    z_sub  = all_z[idx]; ctx_sub = all_ctype[idx]
    pca    = PCA(n_components=2); z_2d = pca.fit_transform(z_sub)
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(len(M.CONTEXTS)):
        mask = ctx_sub == c
        if mask.sum():
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=COLORS[c], label=M.C_LABELS[c], alpha=0.4, s=10)
    ax.set_title(f"PCA of z_t — coloured by context\n{TAG}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(); plt.tight_layout()
    out = fig_path("pca.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_confusion(clf, all_z, all_ctype, labels):
    cm = confusion_matrix(all_ctype, clf.predict(all_z))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_title(f"Context confusion matrix\n{TAG}")
    plt.tight_layout()
    out = fig_path("confusion.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_perdim(acc_per_dim, n_ctx):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(M.LATENT_DIM), acc_per_dim, color="steelblue")
    ax.axhline(1/n_ctx, color="tomato", ls="--", label=f"chance ({1/n_ctx:.2f})")
    ax.set_xticks(range(M.LATENT_DIM))
    ax.set_xlabel("z dimension"); ax.set_ylabel("Decode accuracy")
    ax.set_title(f"Per-dim context decodability\n{TAG}")
    ax.legend(); plt.tight_layout()
    out = fig_path("perdim.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_decodetime(all_z, all_ctype, N_val, n_ctx):
    all_z_seq     = all_z.reshape(N_val, M.SEQ_LEN, M.LATENT_DIM)
    all_ctype_seq = all_ctype.reshape(N_val, M.SEQ_LEN)
    acc_over_t = []
    for t in range(M.SEQ_LEN):
        z_t = all_z_seq[:, t, :]
        c_t = all_ctype_seq[:, t].astype(int)
        acc_over_t.append(
            cross_val_score(LogisticRegression(max_iter=500), z_t, c_t,
                            cv=5, scoring="accuracy").mean()
        )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(acc_over_t, color="steelblue", label="decode accuracy")
    ax.axhline(1/n_ctx, color="gray",   ls="--", label="chance")
    ax.axvline(50,       color="tomato", ls="--", label="avg switch (t≈50)")
    ax.set_xlabel("Timestep t"); ax.set_ylabel("Context decode accuracy")
    ax.set_title(f"Context decodability over time\n{TAG}")
    ax.legend(); ax.set_ylim(0, 1); plt.tight_layout()
    out = fig_path("decodetime.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_decoder_importance(clf):
    decoder_importance = np.linalg.norm(clf.coef_, axis=0)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(M.LATENT_DIM), decoder_importance, color="seagreen")
    ax.set_xticks(range(M.LATENT_DIM))
    ax.set_xlabel("EC neuron"); ax.set_ylabel("||coef|| across context classes")
    ax.set_title(f"Joint decoder weight norm\n{TAG}")
    plt.tight_layout()
    out = fig_path("decoder_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_svd_singular_values(S_sv, n_axes):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(1, n_axes + 1), S_sv[:n_axes], color="steelblue")
    ax.set_xticks(range(1, n_axes + 1))
    ax.set_xlabel("Context axis"); ax.set_ylabel("Singular value")
    ax.set_title(f"Dimensionality of context subspace\n{TAG}")
    plt.tight_layout()
    out = fig_path("svd_singular_values.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_svd_U_heatmap(U, n_axes, labels, n_ctx):
    U_show = U[:, :n_axes]
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(U_show, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_axes))
    ax.set_xticklabels([f"axis {i+1}" for i in range(n_axes)])
    ax.set_yticks(range(n_ctx))
    ax.set_yticklabels(labels)
    for i in range(n_ctx):
        for j in range(n_axes):
            ax.text(j, i, f"{U_show[i, j]:+.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(U_show[i, j]) < 0.5 else "white")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"U — context loadings on SVD axes\n{TAG}")
    plt.tight_layout()
    out = fig_path("svd_U_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_svd_context_axes(Vt, S_sv, n_axes):
    total_var = (S_sv ** 2).sum()
    fig, axes = plt.subplots(1, n_axes, figsize=(4 * n_axes, 3), sharey=True)
    if n_axes == 1:
        axes = [axes]
    for i, axp in enumerate(axes):
        axp.bar(range(M.LATENT_DIM), Vt[i], color="steelblue")
        axp.axhline(0, color="black", lw=0.5)
        axp.set_title(f"Context axis {i+1}\n({S_sv[i]**2/total_var*100:.0f}% var)")
        axp.set_xlabel("EC neuron")
    axes[0].set_ylabel("Vᵀ loading")
    plt.suptitle(f"Which EC neurons encode each context axis\n{TAG}", y=1.02)
    plt.tight_layout()
    out = fig_path("svd_context_axes.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_cb_context_alignment(cb, Vt, n_axes):
    cb_W_z = cb.net[0].weight.detach().cpu().numpy()[:, 4:]   # (CB_HIDDEN, LATENT_DIM)
    alignment        = cb_W_z @ Vt[:n_axes].T
    cb_w_norm        = np.linalg.norm(cb_W_z, axis=1, keepdims=True) + 1e-8
    alignment_normed = np.linalg.norm(alignment, axis=1) / cb_w_norm[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(alignment_normed, bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(alignment_normed.mean(), color="tomato", ls="--",
                    label=f"mean={alignment_normed.mean():.2f}")
    axes[0].set_xlabel("Fraction of CB neuron's z-weight in context subspace")
    axes[0].set_ylabel("CB neurons"); axes[0].set_title("CB alignment with EC context subspace")
    axes[0].legend()
    axis_alignment = np.linalg.norm(alignment, axis=0)[:n_axes]
    axes[1].bar(range(1, n_axes+1), axis_alignment, color="steelblue")
    axes[1].set_xlabel("Context axis"); axes[1].set_ylabel("Total CB weight (L2)")
    axes[1].set_title("CB weight on each EC context axis")
    plt.suptitle(f"Does CB read EC's context subspace?\n{TAG}")
    plt.tight_layout()
    out = fig_path("cb_context_alignment.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_ablation_per_neuron(delta_mse, acc_per_dim):
    sorted_idx = np.argsort(acc_per_dim)[::-1]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(M.LATENT_DIM), delta_mse[sorted_idx], color="steelblue")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(M.LATENT_DIM))
    ax.set_xticklabels([f"z{sorted_idx[i]}\n{acc_per_dim[sorted_idx[i]]:.2f}"
                        for i in range(M.LATENT_DIM)], fontsize=7)
    ax.set_xlabel("z neuron (sorted by decode acc)"); ax.set_ylabel("ΔMSE")
    ax.set_title(f"Per-neuron ablation effect on CB MSE\n{TAG}")
    plt.tight_layout()
    out = fig_path("ablation_per_neuron.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_ablation_progressive(prog_mse, mse_baseline):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(M.LATENT_DIM + 1), prog_mse, color="steelblue", marker="o")
    ax.axhline(mse_baseline, color="gray", ls="--", label=f"baseline={mse_baseline:.4f}")
    ax.set_xlabel("Neurons ablated (highest decode acc first)"); ax.set_ylabel("CB MSE")
    ax.set_title(f"Progressive ablation\n{TAG}"); ax.legend()
    plt.tight_layout()
    out = fig_path("ablation_progressive.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_ctx_subspace_ablation(ctx_prog_mse, mse_baseline, n_axes):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(n_axes + 1), ctx_prog_mse, color="seagreen", marker="o")
    ax.axhline(mse_baseline, color="gray", ls="--", label=f"baseline={mse_baseline:.4f}")
    ax.set_xlabel("Context axes removed"); ax.set_ylabel("CB MSE")
    ax.set_xticks(range(n_axes + 1))
    ax.set_title(f"Context subspace ablation\n{TAG}"); ax.legend()
    plt.tight_layout()
    out = fig_path("ctx_subspace_ablation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_param_encoding_all(all_z, ctx_arrays):
    # ctx_arrays: dict of name -> (1-D target array, color)
    fig, axes = plt.subplots(1, len(ctx_arrays), figsize=(6 * len(ctx_arrays), 3))
    for axp, (name, (y, color)) in zip(axes.flat, ctx_arrays.items()):
        r2_dim = [cross_val_score(Ridge(), all_z[:, d:d+1], y,
                                  cv=5, scoring="r2").mean() for d in range(M.LATENT_DIM)]
        axp.bar(range(M.LATENT_DIM), r2_dim, color=color)
        axp.axhline(0, color="black", lw=0.5)
        axp.set_xlabel("EC neuron"); axp.set_ylabel("R²")
        axp.set_title(f"Per-neuron R² for {name}")
    plt.suptitle(f"Per-neuron parameter encoding (univariate)\n{TAG}", y=1.05)
    plt.tight_layout()
    out = fig_path("param_encoding_all.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_param_one_glm(all_z, ctx_arrays):
    fig, axes = plt.subplots(1, len(ctx_arrays), figsize=(6 * len(ctx_arrays), 3))
    for axp, (name, (y, color)) in zip(axes.flat, ctx_arrays.items()):
        ridge = Ridge(); ridge.fit(all_z, y)
        coefs = np.abs(ridge.coef_)
        axp.bar(range(M.LATENT_DIM), coefs, color=color)
        axp.set_xticks(range(0, M.LATENT_DIM, 2))
        axp.set_xlabel("EC neuron"); axp.set_ylabel("|coef|")
        axp.set_title(f"Joint ridge |coef| for {name}")
        top3 = np.argsort(coefs)[-3:][::-1]
        print(f"  {name}: top-3 by |coef| = {top3.tolist()}  vals = {coefs[top3].round(3).tolist()}")
    plt.suptitle(f"Joint parameter ridge weights\n{TAG}", y=1.05)
    plt.tight_layout()
    out = fig_path("param_one_glm.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_complement_subspace_ablation(cb_mse_with_z, mse_baseline, all_z_seq, Vt, n_axes):
    Vt_ctx = Vt[:n_axes]; P_ctx = Vt_ctx.T @ Vt_ctx
    z_only_ctx  = all_z_seq @ P_ctx
    z_only_comp = all_z_seq - z_only_ctx
    z_zero      = np.zeros_like(all_z_seq)
    vals = {
        f"Full z\n(32-D)":                          mse_baseline,
        f"Context subspace only\n({n_axes}-D)":     cb_mse_with_z(z_only_ctx),
        f"Complement only\n({M.LATENT_DIM - n_axes}-D, no ctx)": cb_mse_with_z(z_only_comp),
        f"z = 0\n(EC silenced)":                    cb_mse_with_z(z_zero),
    }
    print("\nComplement-subspace analysis:")
    for k, v in vals.items():
        print(f"  {k.splitlines()[0]:30s}  MSE = {v:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(vals.keys()), list(vals.values()),
           color=["steelblue", "seagreen", "tomato", "gray"])
    ax.axhline(mse_baseline, color="gray", ls="--", lw=1, label=f"baseline={mse_baseline:.4f}")
    for i, v in enumerate(vals.values()):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("CB MSE"); ax.set_title(f"What does CB need from z?\n{TAG}")
    ax.legend(); plt.tight_layout()
    out = fig_path("complement_subspace_ablation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_state_only_context_control(all_z, all_ctype, va_trans):
    """Bar plot: context R² from z vs. context R² from state-features alone."""
    va_np = va_trans.numpy()
    s_curr = va_np[:, :, 0:2]
    s_prev = np.zeros_like(s_curr); s_prev[:, 1:] = s_curr[:, :-1]
    s_next = va_np[:, :, 4:6]
    X_state = np.hstack([s_curr.reshape(-1, 2),
                         s_prev.reshape(-1, 2),
                         s_next.reshape(-1, 2)])  # (N, 6)
    n_ctx = len(M.CONTEXTS)
    C_onehot = np.eye(n_ctx)[all_ctype]

    r2_z     = cross_val_score(Ridge(), all_z, C_onehot, cv=5, scoring="r2").mean()
    r2_state = cross_val_score(Ridge(), X_state, C_onehot, cv=5, scoring="r2").mean()
    direct   = max(0.0, r2_z - r2_state)
    print(f"\nState-mediated vs direct context decoding:")
    print(f"  z → context             R² = {r2_z:.3f}")
    print(f"  state-only → context    R² = {r2_state:.3f}")
    print(f"  estimated direct gap    R² = {direct:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["z → context\n(full z)", "state → context\n(no z)"],
                  [r2_z, r2_state], color=["steelblue", "tomato"])
    for b, v in zip(bars, [r2_z, r2_state]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:+.2f}",
                ha="center", fontsize=10)
    ax.set_ylabel("R² (Ridge on one-hot context, 5-fold CV)")
    ax.set_title(f"Is context info in z just state-mediated?\n{TAG}")
    plt.tight_layout()
    out = fig_path("state_only_context_control.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_family_decoding(all_z, all_ctype, Vt_six_class):
    """Decode 3-class family (rot/curl/drift) ignoring sign. Builds confusion + SVD figs."""
    # Map 6 contexts → 3 families: 0=rot (C0/C1), 1=curl (C2/C3), 2=drift (C4/C5)
    ctx_to_family = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    family_labels = ["rot", "curl", "drift"]
    all_family = np.array([ctx_to_family[int(c)] for c in all_ctype])

    # Joint decode
    clf_fam = LogisticRegression(max_iter=500)
    acc_fam = cross_val_score(clf_fam, all_z, all_family, cv=5, scoring="accuracy").mean()
    clf_fam.fit(all_z, all_family)
    cm = confusion_matrix(all_family, clf_fam.predict(all_z))
    print(f"\nFamily-only decoding: acc = {acc_fam:.3f} (chance = 0.333)")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=family_labels)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Family decode (3-class, ignore sign)\nacc={acc_fam:.3f} | chance=0.333\n{TAG}")
    plt.tight_layout()
    out = fig_path("family_confusion.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")

    # SVD on family decoder W
    W_fam = clf_fam.coef_                              # (3, 32)
    U_fam, S_fam, Vt_fam = np.linalg.svd(W_fam, full_matrices=False)
    n_axes_fam = int(np.sum(S_fam > 1e-6))
    print(f"  family singular values: {S_fam.round(3)}  | n_axes = {n_axes_fam}")

    # Combined singular values + U heatmap
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(1, n_axes_fam + 1), S_fam[:n_axes_fam], color="steelblue")
    axes[0].set_xticks(range(1, n_axes_fam + 1))
    axes[0].set_xlabel("Family axis"); axes[0].set_ylabel("Singular value")
    axes[0].set_title("Family subspace dimensionality")
    U_show = U_fam[:, :n_axes_fam]
    im = axes[1].imshow(U_show, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(range(n_axes_fam))
    axes[1].set_xticklabels([f"axis {i+1}" for i in range(n_axes_fam)])
    axes[1].set_yticks(range(3)); axes[1].set_yticklabels(family_labels)
    for i in range(3):
        for j in range(n_axes_fam):
            axes[1].text(j, i, f"{U_show[i, j]:+.2f}",
                         ha="center", va="center", fontsize=10,
                         color="black" if abs(U_show[i, j]) < 0.5 else "white")
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title("U — family loadings on SVD axes")
    plt.suptitle(f"Family-only SVD\n{TAG}", y=1.02)
    plt.tight_layout()
    out = fig_path("family_svd.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")

    # Family-axes vs 6-class axes overlap (cosine similarity)
    # Vt_six_class shape: (k, 32). Vt_fam shape: (n_axes_fam, 32). Compute cosine sim matrix.
    n_six = Vt_six_class.shape[0]
    cos = np.zeros((n_axes_fam, n_six))
    for i in range(n_axes_fam):
        for j in range(n_six):
            num   = np.dot(Vt_fam[i], Vt_six_class[j])
            denom = np.linalg.norm(Vt_fam[i]) * np.linalg.norm(Vt_six_class[j]) + 1e-12
            cos[i, j] = num / denom

    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(np.abs(cos), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_six))
    ax.set_xticklabels([f"6c axis {i+1}" for i in range(n_six)])
    ax.set_yticks(range(n_axes_fam))
    ax.set_yticklabels([f"family axis {i+1}" for i in range(n_axes_fam)])
    for i in range(n_axes_fam):
        for j in range(n_six):
            ax.text(j, i, f"{abs(cos[i, j]):.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(cos[i, j]) < 0.5 else "black")
    plt.colorbar(im, ax=ax, label="|cosine similarity|")
    ax.set_title(f"Do family axes match 6-class axes 1 & 5?\n{TAG}")
    plt.tight_layout()
    out = fig_path("family_axis_overlap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_per_neuron_contribution_hist(all_z, clf):
    """Two histograms: per-neuron variance & per-neuron joint-decoder weight norm."""
    var_per_neuron = all_z.var(axis=0)                    # (32,)
    weight_norm    = np.linalg.norm(clf.coef_, axis=0)    # (32,)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(var_per_neuron, bins=15, color="steelblue", edgecolor="white")
    axes[0].axvline(var_per_neuron.mean(), color="tomato", ls="--",
                    label=f"mean={var_per_neuron.mean():.3f}")
    axes[0].set_xlabel("Per-neuron variance (across timesteps)")
    axes[0].set_ylabel("# of EC neurons")
    axes[0].set_title("Distribution of per-neuron activity variance")
    axes[0].legend()

    axes[1].hist(weight_norm, bins=15, color="seagreen", edgecolor="white")
    axes[1].axvline(weight_norm.mean(), color="tomato", ls="--",
                    label=f"mean={weight_norm.mean():.2f}")
    axes[1].set_xlabel("Joint-decoder weight norm ||W[:, d]||")
    axes[1].set_ylabel("# of EC neurons")
    axes[1].set_title("Distribution of per-neuron context contribution")
    axes[1].legend()

    plt.suptitle(f"How is z's variance / context-contribution distributed across neurons?\n{TAG}",
                 y=1.02)
    plt.tight_layout()
    out = fig_path("per_neuron_contribution_hist.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def collect_h_and_cb_hidden(ec, cb, va_trans, device):
    """Collect (z, EC hidden h, CB layer-1 post-ReLU, CB layer-2 post-ReLU) for the val set."""
    all_z, all_h, all_cb_h1, all_cb_h2 = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(va_trans), M.BATCH_SIZE):
            trans = va_trans[i:i+M.BATCH_SIZE].to(device)
            B, T, _ = trans.shape
            zeros2  = torch.zeros(B, 2, device=device)

            h      = ec.init_hidden(B, device)
            z_t    = torch.zeros(B, M.LATENT_DIM, device=device)
            s_pred = zeros2.clone()
            z_seq, h_seq, cb_h1_seq, cb_h2_seq = [], [], [], []

            # CB.net = [Linear, ReLU, Linear, ReLU, Linear];
            # post-ReLU activations: after net[1] = layer 1, after net[3] = layer 2.
            for t in range(T):
                s_t = trans[:, t, :2]
                a_t = trans[:, t, 2:4]
                if t == 0:
                    s_tm1 = zeros2; a_tm1 = zeros2; e_t = zeros2
                else:
                    s_tm1 = trans[:, t-1, :2]
                    a_tm1 = trans[:, t-1, 2:4]
                    e_t   = s_t - s_pred
                ec_input = torch.cat([s_tm1, a_tm1, s_t, e_t], dim=-1)
                z_t, h   = ec.step(ec_input, h)
                cb_input = torch.cat([s_t, a_t, z_t], dim=-1)
                cb_h1    = cb.net[1](cb.net[0](cb_input))
                cb_h2    = cb.net[3](cb.net[2](cb_h1))
                s_pred   = cb(s_t, a_t, z_t)
                z_seq.append(z_t)
                h_seq.append(h)
                cb_h1_seq.append(cb_h1)
                cb_h2_seq.append(cb_h2)

            all_z.append(torch.stack(z_seq,    dim=1).cpu().reshape(B * T, -1))
            all_h.append(torch.stack(h_seq,    dim=1).cpu().reshape(B * T, -1))
            all_cb_h1.append(torch.stack(cb_h1_seq, dim=1).cpu().reshape(B * T, -1))
            all_cb_h2.append(torch.stack(cb_h2_seq, dim=1).cpu().reshape(B * T, -1))
    return (torch.cat(all_z).numpy(),
            torch.cat(all_h).numpy(),
            torch.cat(all_cb_h1).numpy(),
            torch.cat(all_cb_h2).numpy())


def fig_svd_on_population(activity, all_ctype, labels, n_ctx, name, tag_suffix):
    """Helper: fit logistic on `activity → context`, SVD on W, plot singular values + U heatmap.
    `activity` shape (N, D)."""
    clf = LogisticRegression(max_iter=500)
    acc = cross_val_score(clf, activity, all_ctype, cv=5, scoring="accuracy").mean()
    clf.fit(activity, all_ctype)
    W = clf.coef_                              # (n_ctx, D)
    U, S_sv, Vt = np.linalg.svd(W, full_matrices=False)
    n_axes = int(np.sum(S_sv > 1e-6))
    print(f"\n{name}: D={activity.shape[1]} | decode acc = {acc:.3f} | "
          f"S = {S_sv.round(3)} | n_axes = {n_axes}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # singular values
    axes[0].bar(range(1, n_axes + 1), S_sv[:n_axes], color="steelblue")
    axes[0].set_xticks(range(1, n_axes + 1))
    axes[0].set_xlabel("Context axis"); axes[0].set_ylabel("Singular value")
    axes[0].set_title(f"{name} — singular values")
    # U heatmap
    U_show = U[:, :n_axes]
    im = axes[1].imshow(U_show, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(range(n_axes))
    axes[1].set_xticklabels([f"axis {i+1}" for i in range(n_axes)])
    axes[1].set_yticks(range(n_ctx))
    axes[1].set_yticklabels(labels)
    for i in range(n_ctx):
        for j in range(n_axes):
            axes[1].text(j, i, f"{U_show[i, j]:+.2f}",
                         ha="center", va="center", fontsize=8,
                         color="black" if abs(U_show[i, j]) < 0.5 else "white")
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title(f"{name} — U heatmap (context loadings)")
    plt.suptitle(f"GLM + SVD on {name}\nacc={acc:.3f} | D={activity.shape[1]} | {tag_suffix}",
                 y=1.02)
    plt.tight_layout()
    out = fig_path(f"svd_on_{name.lower().replace(' ', '_')}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_pca_state_vs_context(all_z, all_ctype, va_trans):
    """4-panel PCA scatter, same axes — coloured by context, state, velocity, current action."""
    va_np = va_trans.numpy()
    s_curr = va_np[:, :, 0:2].reshape(-1, 2)
    a_curr = va_np[:, :, 2:4].reshape(-1, 2)
    s_prev = np.zeros_like(va_np[:, :, 0:2]); s_prev[:, 1:] = va_np[:, 0:-1, 0:2]
    v_inferred = ((va_np[:, :, 0:2] - s_prev) / M.DT).reshape(-1, 2)

    n_plot = min(3000, len(all_z))
    idx    = np.random.default_rng(0).choice(len(all_z), n_plot, replace=False)
    pca    = PCA(n_components=2)
    z_2d   = pca.fit_transform(all_z[idx])

    n_ctx = len(M.CONTEXTS)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Panel 1: context (categorical)
    for c in range(n_ctx):
        m = all_ctype[idx] == c
        if m.sum():
            axes[0].scatter(z_2d[m, 0], z_2d[m, 1],
                            c=COLORS[c], label=M.C_LABELS[c], alpha=0.4, s=10)
    axes[0].set_title("Coloured by context\n(categorical)")
    axes[0].legend(fontsize=7)

    # Panel 2: state x-coord (continuous)
    sc = axes[1].scatter(z_2d[:, 0], z_2d[:, 1],
                          c=s_curr[idx, 0], cmap="viridis", alpha=0.6, s=10)
    plt.colorbar(sc, ax=axes[1], label="x position")
    axes[1].set_title("Coloured by state x-coordinate\n(continuous — large signal in z)")

    # Panel 3: velocity magnitude (continuous)
    v_mag = np.linalg.norm(v_inferred, axis=1)
    sc = axes[2].scatter(z_2d[:, 0], z_2d[:, 1],
                          c=v_mag[idx], cmap="plasma", alpha=0.6, s=10)
    plt.colorbar(sc, ax=axes[2], label="|V_t|")
    axes[2].set_title("Coloured by velocity magnitude\n(continuous — moderate signal in z)")

    # Panel 4: a_t x (architectural negative control — R²=0 from z)
    sc = axes[3].scatter(z_2d[:, 0], z_2d[:, 1],
                          c=a_curr[idx, 0], cmap="coolwarm", alpha=0.6, s=10)
    plt.colorbar(sc, ax=axes[3], label="a_t (x-component)")
    axes[3].set_title("Coloured by current action a_t\n(should look random — R²=0 from z)")

    for axp in axes:
        axp.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.suptitle(f"PCA of z — coloured by different variables\n{TAG}", y=1.02)
    plt.tight_layout()
    out = fig_path("pca_state_vs_context.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_z_variable_decoding(all_z, all_ctype, va_trans):
    va_np = va_trans.numpy()
    s_curr = va_np[:, :, 0:2]; a_curr = va_np[:, :, 2:4]; s_next = va_np[:, :, 4:6]
    s_prev = np.zeros_like(s_curr); s_prev[:, 1:] = s_curr[:, :-1]
    a_prev = np.zeros_like(a_curr); a_prev[:, 1:] = a_curr[:, :-1]
    v_inferred = (s_curr - s_prev) / M.DT

    def flat(x): return x.reshape(-1, x.shape[-1])

    # context as one-hot (R² on same scale as continuous targets)
    n_ctx = len(M.CONTEXTS)
    C_onehot = np.eye(n_ctx)[all_ctype]

    targets = {
        "s_t":             flat(s_curr),
        "s_{t+1}":         flat(s_next),
        "s_{t-1}":         flat(s_prev),
        "a_t":             flat(a_curr),
        "a_{t-1}":         flat(a_prev),
        "V_t (inferred)":  flat(v_inferred),
        "context (1-hot)": C_onehot,
    }
    print("\nz variable decoding (joint ridge, 5-fold CV R²):")
    r2 = {}
    for name, y in targets.items():
        r2[name] = cross_val_score(Ridge(), all_z, y, cv=5, scoring="r2").mean()
        print(f"  {name:18s}  R² = {r2[name]:+.3f}")
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(r2.keys()); rs = [r2[n] for n in names]
    bars = ax.bar(names, rs, color="steelblue")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("R² (joint ridge from full z, 5-fold CV)")
    ax.set_title(f"What can be decoded from z?\n{TAG}")
    plt.xticks(rotation=25, ha="right")
    for bar, r in zip(bars, rs):
        ax.text(bar.get_x() + bar.get_width()/2, r + 0.02 if r >= 0 else r - 0.05,
                f"{r:+.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = fig_path("z_variable_decoding.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


# ── EC→CB synapse readout analyses ────────────────────────────────────────────
# Question: how does CB read z, and what's represented in the directions it
# reads loudest?
#
# Pipeline (no GLMs / SVDs *after* the projection):
#   1. W = (cb.net[0].weight[:, 4:]).T          shape (32, 256)     EC→CB synapse
#   2. SVD W = U_c · S_c · Vᵀ_c
#      U_c columns (in z-space, R³²) = directions of z sorted by how strongly
#      W transmits each one to CB. "CB's preferred input axes."
#   3. z_proj = z @ U_c                          shape (16000, 32)
#      EC activity rotated into CB's preferred basis. Axes now ordered by CB's
#      read strength.
#   4. Pairwise scatter of z_proj's top-3 axes, coloured by state / velocity /
#      context → visualises what CB reads loudest.
#   5. Causal test: zero out top-k columns of z_proj (CB's loudest reads),
#      rotate back, measure CB MSE → does CB really need its top reads?

def fig_synapse_singular_values(S_c, prefix="synapse",
                                axis_label="CB-preferred",
                                title="Synapse readout strength per axis  (SVD of EC→CB W)"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(S_c) + 1), S_c, color="steelblue")
    ax.set_xlabel(f"{axis_label} axis  (sorted by W singular value)")
    ax.set_ylabel("Singular value of W")
    ax.set_title(f"{title}\n{TAG}")
    plt.tight_layout()
    out = fig_path(f"{prefix}_singular_values.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


_VIEWS = [(25, 30), (25, 120), (25, 210), (25, 300)]


def fig_synapse_top3_axes_context(z_proj, all_ctype, prefix="synapse",
                                  axis_label="CB-preferred"):
    """4-view 3D scatter of projected signal's top-3 axes, coloured by context."""
    n_plot = min(3000, len(z_proj))
    idx    = np.random.default_rng(0).choice(len(z_proj), n_plot, replace=False)
    n_ctx  = len(M.CONTEXTS)
    fig = plt.figure(figsize=(13, 11))
    for v, (elev, azim) in enumerate(_VIEWS):
        ax = fig.add_subplot(2, 2, v + 1, projection="3d")
        for c in range(n_ctx):
            m = all_ctype[idx] == c
            if m.sum():
                ax.scatter(z_proj[idx[m], 0], z_proj[idx[m], 1], z_proj[idx[m], 2],
                           c=COLORS[c],
                           label=M.C_LABELS[c] if v == 0 else None,
                           alpha=0.5, s=6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("axis 1"); ax.set_ylabel("axis 2"); ax.set_zlabel("axis 3")
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=9)
    fig.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 0.96))
    plt.suptitle(f"Top-3 {axis_label} axes — coloured by context\n{TAG}", y=0.99)
    plt.tight_layout()
    out = fig_path(f"{prefix}_top3_axes_context.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_synapse_top3_axes_continuous(z_proj, color_values, cmap, color_label,
                                     title_suffix, fname, axis_label="CB-preferred"):
    """4-view 3D scatter of projected signal's top-3 axes, coloured by a continuous var."""
    n_plot = min(3000, len(z_proj))
    idx    = np.random.default_rng(0).choice(len(z_proj), n_plot, replace=False)
    fig = plt.figure(figsize=(13, 11))
    sc = None
    for v, (elev, azim) in enumerate(_VIEWS):
        ax = fig.add_subplot(2, 2, v + 1, projection="3d")
        sc = ax.scatter(z_proj[idx, 0], z_proj[idx, 1], z_proj[idx, 2],
                        c=color_values[idx], cmap=cmap, alpha=0.6, s=6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("axis 1"); ax.set_ylabel("axis 2"); ax.set_zlabel("axis 3")
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=9)
    fig.colorbar(sc, ax=fig.get_axes(), label=color_label,
                 shrink=0.6, fraction=0.03, pad=0.04)
    plt.suptitle(f"Top-3 {axis_label} axes — coloured by {title_suffix}\n{TAG}",
                 y=0.99)
    out = fig_path(fname)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_synapse_top3_axes_state_xy(z_proj, state_x, state_y, grid_size,
                                   prefix="synapse", axis_label="CB-preferred"):
    """4-view 3D scatter of projected signal's top-3 axes, coloured by (x, y) via RGB
    (x → red, y → green). Reveals 4-corner workspace structure if present."""
    n_plot = min(3000, len(z_proj))
    idx    = np.random.default_rng(0).choice(len(z_proj), n_plot, replace=False)
    r = np.clip(state_x[idx] / grid_size, 0, 1)
    g = np.clip(state_y[idx] / grid_size, 0, 1)
    b = np.zeros_like(r)
    rgb = np.stack([r, g, b], axis=1)

    fig = plt.figure(figsize=(13, 11))
    for v, (elev, azim) in enumerate(_VIEWS):
        ax = fig.add_subplot(2, 2, v + 1, projection="3d")
        ax.scatter(z_proj[idx, 0], z_proj[idx, 1], z_proj[idx, 2],
                   c=rgb, alpha=0.6, s=6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("axis 1"); ax.set_ylabel("axis 2"); ax.set_zlabel("axis 3")
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=9)

    # Discrete corner legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0, 0, 0), markersize=8, label='(0, 0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(1, 0, 0), markersize=8, label=f'({grid_size}, 0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0, 1, 0), markersize=8, label=f'(0, {grid_size})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(1, 1, 0), markersize=8, label=f'({grid_size}, {grid_size})'),
    ]
    fig.legend(handles=legend_handles, title="(x, y) corner",
               fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 0.96))
    plt.suptitle(f"Top-3 {axis_label} axes — coloured by (x, y) via RGB\n{TAG}",
                 y=0.99)
    plt.tight_layout()
    out = fig_path(f"{prefix}_top3_axes_state_xy.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


def fig_synapse_zproj_ablation(cb_mse, z_proj, U_c, N_val, mse_baseline, k_max):
    """Ablate top-k CB-preferred axes of z_proj: zero out z_proj[:, :k],
       rotate back to z, measure CB MSE. Tests whether CB actually needs
       what it reads loudest."""
    z_proj_seq = z_proj.reshape(N_val, M.SEQ_LEN, -1)
    mse_cb_pref = [mse_baseline]
    for k in range(1, k_max + 1):
        z_proj_abl = z_proj_seq.copy(); z_proj_abl[:, :, :k] = 0.0
        mse_cb_pref.append(cb_mse(z_proj_abl @ U_c.T))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(k_max + 1), mse_cb_pref, marker="o", color="steelblue",
            label="Ablate top-k CB-preferred axes  (zero z_proj[:, :k])")
    ax.axhline(mse_baseline, color="gray", ls="--", lw=1,
               label=f"baseline = {mse_baseline:.4f}")
    ax.set_xlabel("Top-k axes removed from z @ U_c")
    ax.set_ylabel("CB MSE")
    ax.set_title(f"Ablation along CB's preferred reading axes\n{TAG}")
    ax.legend(); plt.tight_layout()
    out = fig_path("synapse_zproj_ablation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model + history
    ckpt = torch.load(M.CKPT_PATH, map_location=device, weights_only=False)
    ec = M.EntorhinalCortex().to(device); ec.load_state_dict(ckpt["ec"])
    cb = M.CerebellumCB().to(device);     cb.load_state_dict(ckpt["cb"])
    ec.eval(); cb.eval()
    history = ckpt["history"]
    print(f"Loaded {M.CKPT_PATH}")

    # Regenerate val data + z
    va_trans, va_ctx = M.generate_sequences(M.N_VAL, seed=99)
    all_z, all_ctype = collect_z_and_context(ec, cb, va_trans, va_ctx, device)
    N_val   = len(va_trans)
    n_ctx   = len(M.CONTEXTS)
    labels  = [M.C_LABELS[i] for i in range(n_ctx)]
    os.makedirs(M.FIG_DIR, exist_ok=True)

    # Joint context decoder + SVD (used by many figs)
    clf = LogisticRegression(max_iter=500); clf.fit(all_z, all_ctype)
    W = clf.coef_
    U, S_sv, Vt = np.linalg.svd(W, full_matrices=False)
    n_axes = int(np.sum(S_sv > 1e-6))
    print(f"Singular values: {S_sv.round(3)}  | n_axes (nonzero) = {n_axes}")

    # Univariate per-neuron decode acc (used by perdim + ablation ordering)
    acc_per_dim = np.array([
        cross_val_score(LogisticRegression(max_iter=500), all_z[:, d:d+1],
                        all_ctype, cv=5, scoring="accuracy").mean()
        for d in range(M.LATENT_DIM)
    ])

    cb_mse = make_cb_mse_fn(cb, va_trans, device)
    all_z_seq = all_z.reshape(N_val, M.SEQ_LEN, M.LATENT_DIM)
    mse_baseline = cb_mse(all_z_seq)
    print(f"Baseline CB MSE: {mse_baseline:.4f}")

    # Per-neuron ablation
    delta_mse = np.zeros(M.LATENT_DIM)
    for d in range(M.LATENT_DIM):
        z_abl = all_z_seq.copy(); z_abl[:, :, d] = 0.0
        delta_mse[d] = cb_mse(z_abl) - mse_baseline

    # Progressive ablation by decode acc
    sorted_dims = np.argsort(acc_per_dim)[::-1]
    prog_mse = [mse_baseline]
    for k in range(1, M.LATENT_DIM + 1):
        z_abl = all_z_seq.copy(); z_abl[:, :, sorted_dims[:k]] = 0.0
        prog_mse.append(cb_mse(z_abl))

    # Context subspace ablation (project off top-k axes)
    ctx_prog_mse = [mse_baseline]
    for k in range(1, n_axes + 1):
        Vt_k  = Vt[:k]; Pc_k = Vt_k.T @ Vt_k
        z_abl = all_z_seq - all_z_seq @ Pc_k
        ctx_prog_mse.append(cb_mse(z_abl))

    # Parameter arrays for ridge analyses
    ctx_to_theta = {0: np.pi/4, 1: -np.pi/4, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    ctx_to_gamma = {0: 0.0,     1: 0.0,      2: 0.4, 3: -0.4, 4: 0.0, 5: 0.0}
    ctx_to_g     = {0: 0.0,     1: 0.0,      2: 0.0, 3: 0.0, 4: 0.4, 5: 0.4}
    param_arrays = {
        "θ (rotation angle)": (np.array([ctx_to_theta[int(c)] for c in all_ctype]), "steelblue"),
        "γ (curl strength)":  (np.array([ctx_to_gamma[int(c)] for c in all_ctype]), "tomato"),
        "g (drift strength)": (np.array([ctx_to_g[int(c)]     for c in all_ctype]), "seagreen"),
    }

    # ── generate every figure ──
    fig_training(history)
    fig_pca(all_z, all_ctype)
    fig_confusion(clf, all_z, all_ctype, labels)
    fig_perdim(acc_per_dim, n_ctx)
    fig_decodetime(all_z, all_ctype, N_val, n_ctx)
    fig_decoder_importance(clf)
    fig_svd_singular_values(S_sv, n_axes)
    fig_svd_U_heatmap(U, n_axes, labels, n_ctx)
    fig_svd_context_axes(Vt, S_sv, n_axes)
    fig_cb_context_alignment(cb, Vt, n_axes)
    fig_ablation_per_neuron(delta_mse, acc_per_dim)
    fig_ablation_progressive(prog_mse, mse_baseline)
    fig_ctx_subspace_ablation(ctx_prog_mse, mse_baseline, n_axes)
    fig_param_encoding_all(all_z, param_arrays)
    fig_param_one_glm(all_z, param_arrays)
    fig_complement_subspace_ablation(cb_mse, mse_baseline, all_z_seq, Vt, n_axes)
    fig_z_variable_decoding(all_z, all_ctype, va_trans)
    fig_pca_state_vs_context(all_z, all_ctype, va_trans)
    fig_state_only_context_control(all_z, all_ctype, va_trans)
    fig_family_decoding(all_z, all_ctype, Vt[:n_axes])
    fig_per_neuron_contribution_hist(all_z, clf)

    # GLM + SVD on EC hidden h and CB hidden layer
    _, all_h, all_cb_h1, all_cb_h2 = collect_h_and_cb_hidden(ec, cb, va_trans, device)
    fig_svd_on_population(all_h,     all_ctype, labels, n_ctx, "EC hidden h",   TAG)
    fig_svd_on_population(all_cb_h1, all_ctype, labels, n_ctx, "CB layer-1",    TAG)
    fig_svd_on_population(all_cb_h2, all_ctype, labels, n_ctx, "CB layer-2",    TAG)

    # ── EC→CB synapse readout: rotate z into CB's preferred basis, visualize ──
    W_synapse = cb.net[0].weight.detach().cpu().numpy()[:, 4:].T   # (32, 256)
    U_c_syn, S_c_syn, _ = np.linalg.svd(W_synapse, full_matrices=False)
    print(f"\nSynapse W shape: {W_synapse.shape}")
    print(f"Synapse singular values (top-5): {S_c_syn[:5].round(3)}")

    z_proj = all_z @ U_c_syn                                       # (16000, 32)

    fig_synapse_singular_values(S_c_syn)

    # Top-3 CB-preferred axes — pairwise scatters, coloured per variable
    va_np    = va_trans.numpy()
    s_curr_x = va_np[:, :, 0].reshape(-1)
    s_curr_y = va_np[:, :, 1].reshape(-1)
    s_prev   = np.zeros_like(va_np[:, :, 0:2]); s_prev[:, 1:] = va_np[:, 0:-1, 0:2]
    v_mag    = np.linalg.norm((va_np[:, :, 0:2] - s_prev) / M.DT, axis=-1).reshape(-1)

    fig_synapse_top3_axes_context(z_proj, all_ctype)
    fig_synapse_top3_axes_continuous(z_proj, s_curr_x, "viridis", "state x",
                                     "state x-coordinate",
                                     "synapse_top3_axes_state.png")
    fig_synapse_top3_axes_state_xy(z_proj, s_curr_x, s_curr_y, M.GRID_SIZE)
    fig_synapse_top3_axes_continuous(z_proj, v_mag, "plasma", "|V_t|",
                                     "velocity magnitude",
                                     "synapse_top3_axes_velocity.png")

    fig_synapse_zproj_ablation(cb_mse, z_proj, U_c_syn, N_val,
                               mse_baseline, k_max=M.LATENT_DIM)

    # ── EC readout: rotate h_ec into the readout's preferred basis, visualize ──
    # Pipeline (analogous to z @ U_cb above, one stage upstream):
    #   1. W_ec = ec.readout[0].weight.T          shape (256, 32)
    #   2. SVD(W_ec) → U_ec (256, 32), S_ec (32,), Vt_ec (32, 32)
    #   3. h_proj = h_ec @ U_ec                   shape (16000, 32)
    #      EC hidden activity rotated into readout-preferred basis.
    #   4. Plot top-3 columns of h_proj coloured by state_xy / |V| / context.
    W_ec_readout = ec.readout[0].weight.detach().cpu().numpy().T   # (256, 32)
    U_ec, S_ec, _ = np.linalg.svd(W_ec_readout, full_matrices=False)
    print(f"\nEC readout W shape: {W_ec_readout.shape}")
    print(f"EC readout singular values (top-5): {S_ec[:5].round(3)}")

    h_proj = all_h @ U_ec                                          # (16000, 32)

    fig_synapse_singular_values(
        S_ec, prefix="ec_readout", axis_label="readout-preferred",
        title="EC readout strength per axis  (SVD of h_ec→z W)")
    fig_synapse_top3_axes_context(h_proj, all_ctype,
                                  prefix="ec_readout",
                                  axis_label="readout-preferred")
    fig_synapse_top3_axes_continuous(h_proj, s_curr_x, "viridis", "state x",
                                     "state x-coordinate",
                                     "ec_readout_top3_axes_state.png",
                                     axis_label="readout-preferred")
    fig_synapse_top3_axes_state_xy(h_proj, s_curr_x, s_curr_y, M.GRID_SIZE,
                                   prefix="ec_readout",
                                   axis_label="readout-preferred")
    fig_synapse_top3_axes_continuous(h_proj, v_mag, "plasma", "|V_t|",
                                     "velocity magnitude",
                                     "ec_readout_top3_axes_velocity.png",
                                     axis_label="readout-preferred")

    # ── CB layer-1 and layer-2: PCA top-3 of post-ReLU activations ────────────
    # No weight-matrix SVD here — purely a data-driven PCA of each layer's
    # 256-D post-ReLU activations. Top-3 PCs are the directions of maximum
    # variance in the layer's activity.
    for layer_name, activity in [("cb_layer1", all_cb_h1),
                                 ("cb_layer2", all_cb_h2)]:
        X       = activity - activity.mean(0, keepdims=True)
        _, S_pc, Vt_pc = np.linalg.svd(X, full_matrices=False)
        pcs     = X @ Vt_pc.T                                    # (16k, 256)
        print(f"\n{layer_name} PCA top-3 singular values: {S_pc[:3].round(2)}")

        prefix = layer_name
        axlbl  = f"{layer_name} PC"
        fig_synapse_top3_axes_context(pcs, all_ctype,
                                      prefix=prefix, axis_label=axlbl)
        fig_synapse_top3_axes_continuous(pcs, s_curr_x, "viridis", "state x",
                                         "state x-coordinate",
                                         f"{prefix}_top3_axes_state.png",
                                         axis_label=axlbl)
        fig_synapse_top3_axes_state_xy(pcs, s_curr_x, s_curr_y, M.GRID_SIZE,
                                       prefix=prefix, axis_label=axlbl)
        fig_synapse_top3_axes_continuous(pcs, v_mag, "plasma", "|V_t|",
                                         "velocity magnitude",
                                         f"{prefix}_top3_axes_velocity.png",
                                         axis_label=axlbl)

    print(f"\nAll figures saved to {M.FIG_DIR}/")


if __name__ == "__main__":
    main()

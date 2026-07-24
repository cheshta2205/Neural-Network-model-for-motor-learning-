"""
replot_mtl_geometry.py
──────────────────────
Titleless remakes for the document:
  confusion_notitle.png            — symmetric, row-normalized confusion (z_ec, magma)
  mtl_dotproduct_mds_notitle.png   — decoder-weight cosine heatmap + MDS (MTL/EC_hidden)

Reproduces the existing conventions (ECtoCB_train_MSE_12C confusion; MLP_12c dot-product),
just without titles.
"""

import os
import importlib.util
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.manifold import MDS

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mlp12c", os.path.join(HERE, "MLP_12c.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
M = m.M

device = "cpu"; n_seqs = 500
NCTX = len(M.CONTEXTS)
C_LABELS = M.C_LABELS; COLORS = M.COLORS
labels = [C_LABELS[i] for i in range(NCTX)]
FIG_DIR = M.FIG_DIR

base = Path(M.CKPT_PATH)
ckpt_path = [f for f in base.iterdir() if f.is_file() and "_seed" in f.name][0]
ckpt = torch.load(ckpt_path, map_location=device)
ec = M.EntorhinalCortex().to(device); ec.load_state_dict(ckpt["ec"]); ec.eval()
cb = M.CerebellumCB().to(device);     cb.load_state_dict(ckpt["cb"]); cb.eval()

va_trans, va_ctx = M.generate_sequences(n_seqs, seed=99)
z, c, s, a, ec_r, cb_1, cb_2 = m.collect_z_state_action(ec, cb, va_trans, va_ctx, device)
print("collected reps", flush=True)

# ── Fig 3: confusion (z_ec) ──────────────────────────────────────────────────
zc = StandardScaler().fit_transform(z)
clf = LogisticRegression(max_iter=500).fit(zc, c)
cm = confusion_matrix(c, clf.predict(zc)).astype(float)
rs = cm.sum(axis=1, keepdims=True)
cmn = np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0)
sym = 0.5 * (cmn + cmn.T)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(sym, cmap="magma", vmin=0, vmax=1)
ax.set_xticks(range(NCTX)); ax.set_yticks(range(NCTX))
ax.set_xticklabels(labels, rotation=30, ha="right"); ax.set_yticklabels(labels)
ax.set_xlabel("context"); ax.set_ylabel("context")
for i in range(NCTX):
    for j in range(NCTX):
        ax.text(j, i, f"{sym[i, j]:.2f}", ha="center", va="center",
                color="white" if sym[i, j] < 0.5 else "black", fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
out1 = os.path.join(FIG_DIR, "confusion_notitle.png")
plt.savefig(out1, bbox_inches="tight", dpi=200); plt.close()
print(f"wrote {out1}", flush=True)

# ── Fig 4: decoder-weight cosine heatmap + MDS (MTL = EC_hidden) ──────────────
ecs = StandardScaler().fit_transform(ec_r)
clf2 = LogisticRegression(max_iter=500).fit(ecs, c)
W = clf2.coef_
U, S_sv, Vt = np.linalg.svd(W, full_matrices=False)
rank = NCTX // 2
W_tilde = U[:, :rank] @ np.diag(S_sv[:rank]) @ Vt[:rank]
W_normed = W_tilde / (np.linalg.norm(W_tilde, axis=1, keepdims=True) + 1e-8)
dp = W_normed @ W_normed.T
dist = 1 - dp
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
          n_init=8, normalized_stress="auto")
coords = mds.fit_transform(dist)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
g = ax[0].imshow(dp, cmap="RdBu", vmin=-1, vmax=1)
fig.colorbar(g, ax=ax[0], pad=0.05, fraction=0.05, label="Dot Product")
ax[0].set_xticks(range(NCTX)); ax[0].set_yticks(range(NCTX))
ax[0].set_xticklabels(labels, rotation=90); ax[0].set_yticklabels(labels)
for i in range(NCTX):
    ax[1].scatter(coords[i, 0], coords[i, 1], s=180, color=COLORS[i],
                  edgecolor="black", lw=0.7)
    ax[1].annotate(labels[i], (coords[i, 0], coords[i, 1]),
                   textcoords="offset points", xytext=(8, 6), fontsize=6, color=COLORS[i])
ax[1].set_xlabel("MDS dim 1"); ax[1].set_ylabel("MDS dim 2")
ax[1].set_aspect("equal", adjustable="datalim"); ax[1].grid(alpha=0.3)
fig.tight_layout()
out2 = os.path.join(FIG_DIR, "mtl_dotproduct_mds_notitle.png")
fig.savefig(out2, bbox_inches="tight", dpi=200); plt.close()
print(f"wrote {out2}", flush=True)
print("done", flush=True)

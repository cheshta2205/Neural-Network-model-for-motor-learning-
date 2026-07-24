"""
make_context_3angle_section1.py — static 3-angle PNG of Section-1 MTL EC_hidden
projected onto the context-decoder's top-3 SVD axes gifs cant go into paper so i do this 
to give a full view of how reps in MTL look like. Colored by the 12 contexts.

Run: python make_context_3angle_section1.py
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import ECtoCB_train_MSE_12C as M

SEED_CKPT = 0
N_VAL     = 200
SUB       = 2500
OUT       = "Paper/neural_activity_comparisons/non_sro/section1_EChidden_context_subspace_3angle.png"
NCTX      = len(M.CONTEXTS)
ANGLES    = [(20, 30), (20, 120), (20, 210)]


def main():
    device = "cpu"
    ck = torch.load(M.seeded_ckpt_path(SEED_CKPT), map_location=device)
    ec = M.EntorhinalCortex().to(device); ec.load_state_dict(ck["ec"]); ec.eval()
    cb = M.CerebellumCB().to(device);     cb.load_state_dict(ck["cb"]); cb.eval()
    va_trans, va_ctx = M.generate_sequences(N_VAL, seed=99)
    with torch.no_grad():
        z, _, reps = M.run_sequence(ec, cb, va_trans.to(device), device,
                                    return_representations=True)
    H = reps["EC_hidden"].cpu().numpy().reshape(-1, M.EC_HIDDEN)
    c = va_ctx[:, :, 2].numpy().astype(int).reshape(-1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(c), min(SUB, len(c)), replace=False)
    Hs, cs = H[idx], c[idx]

    clf = LogisticRegression(max_iter=500).fit(Hs, cs)
    _, _, Vt = np.linalg.svd(clf.coef_, full_matrices=False)
    proj = Hs @ Vt[:3].T

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig = plt.figure(figsize=(16, 5.6))
    for k, (elev, azim) in enumerate(ANGLES):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for ci in range(NCTX):
            m = cs == ci
            if m.any():
                ax.scatter(proj[m, 0], proj[m, 1], proj[m, 2], s=6, alpha=0.45,
                           color=M.COLORS[ci], label=M.C_LABELS[ci] if k == 0 else None)
        ax.set_xlabel("ctx ax1"); ax.set_ylabel("ctx ax2"); ax.set_zlabel("ctx ax3")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"view {k+1} (azim={azim}°)")
    fig.legend(loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Section-1 MTL EC_hidden on context-decoder axes (3 angles)", fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

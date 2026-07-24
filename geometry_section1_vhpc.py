"""
geometry_section1_vhpc.py — vHPC (Science 2024, "Hidden state inference requires
abstract contextual representations in ventral hippocampus") Figure-4 analog on the
Section-1 MTL->CB model (ECtoCB_train_MSE_12C.py). MTL hidden state = EC GRU hidden
(256-D) ~ vCA1 population.

Panels laid out to mirror the consolidated paper Fig 4 (A/B/C/D):- check if mtl neural reps match 
that of ventral Hippocampus during non sro task. 

  A (4F-H) geometry   : EC_hidden on context-decoder axes (supervised; NOT PCA)
  B (4K)   decode     : context (12-way) / family (3-way) decoding bars
  C (4L-M) abstraction: family-across-magnitude CCGP vs # MTL units (data vs shuffle)-can c be decoded even if its not trained on exactly that stimuli 
  D (4E)   context cells: eta^2 context-selectivity histogram

Also writes the rotating 3D gif and
3-angle PNG of the context-subspace geometry (check in PAPER folder for gifs -- supplementary). The inference-lock-in
(decode vs trials-since-switch) is no longer in the main figure (no Fig-4 counterpart).

Run: python geometry_section1_vhpc.py        (set MAKE_GIF=False to skip the slow gif)
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import ECtoCB_train_MSE_12C as M

SEED_CKPT = 0
N_VAL     = 200
SUB       = 9000          # subsample trials for decoding
SUB_CURVE = 4000          # subsample trials for the acc-vs-#units curve
SUB3D     = 2500          # subsample for 3D gif / 3-angle
MAKE_GIF  = True
OUTDIR    = "Paper/neural_activity_comparisons/non_sro"
OUT_FIG   = f"{OUTDIR}/section1_mtl_vhpc_fig4"          # .png + .svg appended
OUT_GIF   = f"{OUTDIR}/section1_EChidden_context_subspace.gif"
OUT_3ANG  = f"{OUTDIR}/section1_EChidden_context_subspace_3angle.png"
NCTX      = len(M.CONTEXTS)                       # 12
FAMILY_OF = {c[6]: c[8] for c in M.CONTEXTS}      # c_id -> family (0 rot,1 curl,2 drift)
ANGLES    = [(20, 30), (20, 120), (20, 210)]

LARGE = [0, 2, 4, 6, 8, 10]; SMALL = [1, 3, 5, 7, 9, 11]     # magnitude split
POS   = [0, 1, 4, 5, 8, 9];  NEG   = [2, 3, 6, 7, 10, 11]    # sign split
UNIT_COUNTS = [2, 5, 10, 25, 50, 100, 256]


def trials_since_switch(cid):
    N, T = cid.shape
    tss = np.zeros((N, T), dtype=int)
    for n in range(N):
        k = 0
        for t in range(1, T):
            k = 0 if cid[n, t] != cid[n, t - 1] else k + 1
            tss[n, t] = k
    return tss


def context_axes(H, c, k=3):
    """Top-k SVD axes of the context-decoder weights (context-discriminating dirs)."""
    clf = LogisticRegression(max_iter=500).fit(H, c)
    _, _, Vt = np.linalg.svd(clf.coef_, full_matrices=False)
    return Vt[:k]


def ccgp_fam_subset(H, cs, fs, units):
    """Family-across-magnitude CCGP using only `units` of the population."""
    Hu = H[:, units]
    a = []
    for tr, te in ((LARGE, SMALL), (SMALL, LARGE)):
        trm = np.isin(cs, tr); tem = np.isin(cs, te)
        clf = LogisticRegression(max_iter=300).fit(Hu[trm], fs[trm])
        a.append((clf.predict(Hu[tem]) == fs[tem]).mean())
    return np.mean(a)


def make_rotating_gif(proj, c, path, title):
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    for ci in range(NCTX):
        m = c == ci
        if m.any():
            ax.scatter(proj[m, 0], proj[m, 1], proj[m, 2], s=6, alpha=0.45,
                       color=M.COLORS[ci], label=M.C_LABELS[ci])
    ax.set_xlabel("ctx ax1"); ax.set_ylabel("ctx ax2"); ax.set_zlabel("ctx ax3")
    ax.set_title(title); ax.legend(fontsize=6, ncol=2, loc="upper left")
    FuncAnimation(fig, lambda i: ax.view_init(elev=20, azim=i * 10) or (),
                  frames=36, interval=120).save(path, writer=PillowWriter(fps=10))
    plt.close(); print(f"wrote {path}", flush=True)


def make_3angle_png(proj, c, path, title):
    fig = plt.figure(figsize=(16, 5.6))
    for k, (elev, azim) in enumerate(ANGLES):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for ci in range(NCTX):
            m = c == ci
            if m.any():
                ax.scatter(proj[m, 0], proj[m, 1], proj[m, 2], s=6, alpha=0.45,
                           color=M.COLORS[ci], label=M.C_LABELS[ci] if k == 0 else None)
        ax.set_xlabel("ctx ax1"); ax.set_ylabel("ctx ax2"); ax.set_zlabel("ctx ax3")
        ax.view_init(elev=elev, azim=azim); ax.set_title(f"view {k+1} (azim={azim}°)")
    fig.legend(loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"wrote {path}", flush=True)


def main():
    device = "cpu"
    ck = torch.load(M.seeded_ckpt_path(SEED_CKPT), map_location=device)
    ec = M.EntorhinalCortex().to(device); ec.load_state_dict(ck["ec"]); ec.eval()
    cb = M.CerebellumCB().to(device);     cb.load_state_dict(ck["cb"]); cb.eval()
    print(f"loaded {M.seeded_ckpt_path(SEED_CKPT)}", flush=True)

    va_trans, va_ctx = M.generate_sequences(N_VAL, seed=99)
    with torch.no_grad():
        _, _, reps = M.run_sequence(ec, cb, va_trans.to(device), device,
                                    return_representations=True)
    H = reps["EC_hidden"].cpu().numpy()
    N, T, D = H.shape
    cid = va_ctx[:, :, 2].numpy().astype(int)
    fam = np.vectorize(FAMILY_OF.get)(cid)
    tss = trials_since_switch(cid)

    Hf = H.reshape(N * T, D); cidf = cid.reshape(-1); famf = fam.reshape(-1)
    tssf = tss.reshape(-1)
    rng = np.random.default_rng(0)
    idx = rng.choice(N * T, min(SUB, N * T), replace=False)
    Hs, cs, fs = Hf[idx], cidf[idx], famf[idx]

    # ---- B: decoding
    dec_ctx = cross_val_score(LogisticRegression(max_iter=500), Hs, cs, cv=4).mean()
    dec_fam = cross_val_score(LogisticRegression(max_iter=500), Hs, fs, cv=4).mean()
    late = tssf >= 15
    Hl, cl = Hf[late], cidf[late]
    li = rng.choice(len(Hl), min(SUB, len(Hl)), replace=False)
    dec_ctx_late = cross_val_score(LogisticRegression(max_iter=500), Hl[li], cl[li], cv=4).mean()

    # ---- C: abstraction — family-across-magnitude CCGP vs # units 
    ci = rng.choice(len(Hs), min(SUB_CURVE, len(Hs)), replace=False)
    Hc, csc, fsc = Hs[ci], cs[ci], fs[ci]
    curve, curve_sh = [], []
    for nu in UNIT_COUNTS:
        v, vsh = [], []
        for _ in range(3):
            u = rng.choice(D, min(nu, D), replace=False)
            v.append(ccgp_fam_subset(Hc, csc, fsc, u))
            vsh.append(ccgp_fam_subset(Hc, csc, rng.permutation(fsc), u))
        curve.append(np.mean(v)); curve_sh.append(np.mean(vsh))
    ccgp_full = curve[-1]
    print(f"decode ctx={dec_ctx:.3f} (locked-in {dec_ctx_late:.3f}) fam={dec_fam:.3f} | "
          f"family-across-mag CCGP (full pop)={ccgp_full:.3f} (chance 0.33)", flush=True)

    # ---- D: context-selective units ? - neurons tuned to C? 
    grand = Hs.mean(0); ss_tot = ((Hs - grand) ** 2).sum(0); ss_bet = np.zeros(D)
    for c in range(NCTX):
        m = cs == c
        if m.sum():
            ss_bet += m.sum() * (Hs[m].mean(0) - grand) ** 2
    eta2 = ss_bet / (ss_tot + 1e-9)
    THRESH = 0.10
    frac_sel = (eta2 > THRESH).mean()
    print(f"context-selective units (eta2>{THRESH}): {frac_sel*100:.0f}%  (vHPC ~30%)", flush=True)

    # ---- A: context-decoder axes (geometry)
    Vt_ctx = context_axes(Hs, cs, k=3)
    pidx = rng.choice(len(Hs), min(SUB3D, len(Hs)), replace=False)
    proj3 = Hs[pidx] @ Vt_ctx.T
    cproj = cs[pidx]

    # ---- figure (A/B/C/D mirroring the paper)
    os.makedirs(OUTDIR, exist_ok=True)
    fig, axd = plt.subplot_mosaic([["A", "B"], ["C", "D"]], figsize=(13, 9))
    fig.suptitle("Section-1 MTL (EC_hidden) — vHPC Fig-4 analog (Science 2024)", fontsize=13)

    # A — geometry
    ax = axd["A"]
    for c in range(NCTX):
        m = cproj == c
        if m.any():
            ax.scatter(proj3[m, 0], proj3[m, 1], s=8, alpha=0.45,
                       color=M.COLORS[c], label=M.C_LABELS[c])
    ax.set_title("A — geometry: EC_hidden on context-decoder axes (4F-H)")
    ax.set_xlabel("ctx axis 1"); ax.set_ylabel("ctx axis 2")
    ax.legend(fontsize=6, ncol=2, loc="best")

    # B — decoding bars
    ax = axd["B"]
    bars = {"context": dec_ctx, "context\n(locked-in)": dec_ctx_late, "family": dec_fam}
    ax.bar(range(3), list(bars.values()), color=["steelblue", "steelblue", "seagreen"])
    ax.axhline(1 / NCTX, color="gray", ls="--", lw=1, label=f"ctx chance ({1/NCTX:.2f})")
    ax.axhline(1 / 3, color="gray", ls=":", lw=1, label="fam chance (0.33)")
    ax.set_xticks(range(3)); ax.set_xticklabels(list(bars.keys()), fontsize=9)
    ax.set_ylim(0, 1.02); ax.set_ylabel("decoding accuracy")
    ax.set_title("B — decode context / family (4K)"); ax.legend(fontsize=8)

    # C — abstraction: CCGP vs # units (data vs shuffle)
    ax = axd["C"]
    ax.plot(UNIT_COUNTS, curve, color="#d62728", marker="o", ms=5, label="data")
    ax.plot(UNIT_COUNTS, curve_sh, color="gray", marker="o", ms=5, label="shuffle")
    ax.axhline(1 / 3, color="gray", ls="--", lw=1, label="chance (0.33)")
    ax.set_xlabel("# MTL units"); ax.set_ylabel("family CCGP (across-magnitude)")
    ax.set_title("C — abstraction: CCGP vs # units (4L-M)"); ax.legend(fontsize=8)

    # D — context cells (eta^2)
    ax = axd["D"]
    ax.hist(eta2, bins=30, color="slateblue", edgecolor="white")
    ax.axvline(THRESH, color="tomato", ls="--", label=f"thresh={THRESH}")
    ax.set_xlabel("context selectivity (eta^2) per unit"); ax.set_ylabel("# MTL units")
    ax.set_title(f"D — context cells: {frac_sel*100:.0f}% selective (4E; vHPC ~30%)")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_FIG + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(OUT_FIG + ".svg", bbox_inches="tight")            # Illustrator-editable
    plt.close()
    print(f"wrote {OUT_FIG}.png and {OUT_FIG}.svg", flush=True)

    # ---- supplementary 3D context-subspace views
    make_3angle_png(proj3, cproj, OUT_3ANG,
                    "Section-1 MTL EC_hidden on context-decoder axes (3 angles)")
    if MAKE_GIF:
        make_rotating_gif(proj3, cproj, OUT_GIF,
                          "MTL EC_hidden on context-decoder axes (Section 1)")


if __name__ == "__main__":
    main()

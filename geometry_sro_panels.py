"""
geometry_sro_panels.py — full Bernardi-style A-E figure for the MTL population (h_t)
of the SRO context-inference model, in BOTH epochs:

  Fig 5 (post-stim) : decode the CURRENT trial's variables from h_t
  Fig 3 (pre-stim)  : decode the PREVIOUS trial's variables from h_t

Panels (matching Bernardi et al. 2020 Fig 3/5):
to compare if MTL reps in our model match the hippocampus neural recordings in the Bernardi paper. 
Run:  python geometry_sro_panels.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"     # keep text editable in Illustrator
plt.rcParams["font.size"] = 12            # bigger base font

import geometry_sro as G                       # reuse the measure functions
from context_inference_sro_rpe import (
    load, generate_sequences, trials_since_switch, N_STIM,
)

N_SEQS = 300
SEED   = 999
M_COND = 60               # trials per condition (subsample)
KMAX   = 15               # max trials-since-switch for panels C/D
OUTDIR = "Paper/neural_activity_comparisons/sro"
COLORS = G.COLORS


# ---- rollout keeping (N, T) structure
def rollout_h(ec, cb, stim, ctx, Ostar, rstar, device):
    B, T = stim.shape
    x_oh = F.one_hot(stim, N_STIM).float().to(device)
    rstar_d = rstar.to(device)
    h = ec.init_hidden(B, device)
    x_prev = torch.zeros(B, N_STIM, device=device)
    delta_pr = torch.zeros(B, 1, device=device)
    H = []
    with torch.no_grad():
        for t in range(T):
            x_t = x_oh[:, t]
            ec_in = torch.cat([x_prev, x_t, delta_pr], dim=-1)
            h = ec.cell(ec_in, h)
            z_t = ec.readout(h)
            cb_h = cb.trunk(torch.cat([x_t, z_t], dim=-1))
            r_pred = cb.reward_head(cb_h)
            H.append(h)
            delta_pr = (rstar_d[:, t:t + 1] - r_pred).detach()
            x_prev = x_t
    return torch.stack(H, 1).cpu().numpy()       # (B, T, 256)


# ---- epoch labels: cond id (ctx*4+stim) per trial, current or previous trial
def cond_labels(stim, ctx, shift_prev=False):
    cond = ctx * 4 + stim                        # (N,T)
    valid = np.ones_like(cond, dtype=bool)
    if shift_prev:
        cond = np.roll(cond, 1, axis=1)
        valid[:, 0] = False                      # t=0 has no previous trial
    return cond, valid


# measures over the 35 dichotomies (panels A/B/E)
def analyze_pooled(X, cond, rng):
    Xs, cs = G.subsample(X, cond, m=M_COND, rng=rng)
    cond_means = {c: Xs[cs == c].mean(0) for c in range(8)}
    dichs = G.all_dichotomies()
    named = G.named_dichotomies()
    name_of = {}
    for nm, (P, N) in named.items():
        sP = frozenset(P)
        for i, (dP, dN) in enumerate(dichs):
            if frozenset(dP) == sP or frozenset(dN) == sP:
                name_of[i] = nm
    dec = np.array([G.decode_accuracy(Xs, cs, P) for P, _ in dichs])
    cc  = np.array([G.ccgp(Xs, cs, P, N) for P, N in dichs])
    ps  = np.array([G.parallelism(cond_means, P, N) for P, N in dichs])
    sd = dec.mean()
    null = []
    for _ in range(20):
        perm = rng.permutation(cs)
        null.append(np.mean([G.decode_accuracy(Xs, perm, P) for P, _ in dichs[:10]]))
    return {"dec": dec, "ccgp": cc, "ps": ps, "sd": sd,
            "null": np.array(null), "name_of": name_of}


# CCGP / PS vs trials-since-switch (panels C/D), for the 3 named variables
def over_tss(X, cond, valid, tss, rng):
    named = G.named_dichotomies()
    out = {nm: {"k": [], "ccgp": [], "ps": []} for nm in named}
    for k in range(KMAX + 1):
        mask = (tss.reshape(-1) == k) & valid.reshape(-1)
        if mask.sum() < 80:
            continue
        Xs, cs = G.subsample(X[mask], cond[mask], m=M_COND, rng=rng)
        if len(set(cs.tolist())) < 8:
            continue
        cond_means = {c: Xs[cs == c].mean(0) for c in range(8)}
        for nm, (P, N) in named.items():
            out[nm]["k"].append(k)
            out[nm]["ccgp"].append(G.ccgp(Xs, cs, P, N))
            out[nm]["ps"].append(G.parallelism(cond_means, P, N))
    return out


# ---- plot one full A-E figure
def plot_panels(res, tss_res, title, fname):
    os.makedirs(OUTDIR, exist_ok=True)
    fig, axd = plt.subplot_mosaic([["A", "B", "E"], ["C", "D", "E"]],
                                  figsize=(15, 8))
    fig.suptitle(title, fontsize=14, y=0.99)

    def hl(ax, vals, x):                       # overlay named-variable dots (with halo)
        for i, nm in res["name_of"].items():
            ax.scatter(x, vals[i], s=190, color="white", zorder=6)          # white halo
            ax.scatter(x, vals[i], color=COLORS[nm], s=120, edgecolors="k",
                       lw=1.1, zorder=7)

    # A: decoding (open) + CCGP (filled)
    ax = axd["A"]; ax.axhline(0.5, color="gray", ls="--", lw=1)
    n = len(res["dec"])
    j = (np.random.default_rng(1).random(n) - 0.5) * 0.3
    ax.scatter(0 + j, res["dec"], facecolors="none", edgecolors="0.5", s=24, lw=0.8)
    ax.scatter(1 + j, res["ccgp"], color="0.55", s=22)
    hl(ax, res["dec"], 0); hl(ax, res["ccgp"], 1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["decoding", "CCGP"])
    loA = min(res["dec"].min(), res["ccgp"].min())
    ax.set_ylim(loA - 0.05, 1.03); ax.set_ylabel("accuracy"); ax.set_title("A — decoding & CCGP")

    # B: parallelism score
    ax = axd["B"]; ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.scatter(0 + j, res["ps"], color="0.6", s=22); hl(ax, res["ps"], 0)
    ax.set_xticks([0]); ax.set_xticklabels(["PS"]); ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel("parallelism score"); ax.set_title("B — parallelism score")

    # C: CCGP vs trials-since-switch
    ax = axd["C"]; ax.axhline(0.5, color="gray", ls="--", lw=1)
    for nm in tss_res:
        d = tss_res[nm]
        ax.plot(d["k"], d["ccgp"], color=COLORS[nm], lw=2, marker="o", ms=4, label=nm)
    ax.set_xlabel("trials since switch"); ax.set_ylabel("CCGP")
    allC = [v for nm in tss_res for v in tss_res[nm]["ccgp"]]
    ax.set_ylim((min(allC) if allC else 0.2) - 0.05, 1.03)
    ax.set_title("C — CCGP over time"); ax.legend(fontsize=8)

    # D: PS vs trials-since-switch
    ax = axd["D"]; ax.axhline(0.0, color="gray", ls="--", lw=1)
    for nm in tss_res:
        d = tss_res[nm]
        ax.plot(d["k"], d["ps"], color=COLORS[nm], lw=2, marker="o", ms=4)
    ax.set_xlabel("trials since switch"); ax.set_ylabel("parallelism score")
    ax.set_title("D — PS over time")

    # E: shattering dim vs null + named CCGP
    ax = axd["E"]; ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    ax.bar(0, res["sd"], width=0.5, color="0.75", edgecolor="k")
    ax.scatter([0] * len(res["null"]), res["null"], color="crimson", s=10, zorder=5,
               label="shuffle null")
    for i, nm in res["name_of"].items():
        ax.scatter(0.5, res["ccgp"][i], color=COLORS[nm], s=90, edgecolors="k", lw=0.7)
    ax.set_xticks([0, 0.5]); ax.set_xticklabels(["SD", "CCGP\n(named)"])
    loE = min(res["sd"], res["null"].min(), min(res["ccgp"][i] for i in res["name_of"]))
    ax.set_ylim(loE - 0.05, 1.03); ax.set_ylabel("decoding / CCGP")
    ax.set_title("E — shattering dim. & abstraction"); ax.legend(fontsize=8)

    handles = [plt.Line2D([0], [0], marker="o", ls="", mec="k", mfc=COLORS[k], ms=9,
               label=k) for k in COLORS]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 0.955))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    base = os.path.join(OUTDIR, fname.replace(".png", ""))
    plt.savefig(base + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(base + ".svg", bbox_inches="tight")         # Illustrator-editable
    plt.close()
    print(f"  wrote {base}.png + .svg", flush=True)


def main():
    rng = np.random.default_rng(SEED)
    ec, cb, device = load()
    stim, ctx, Ostar, rstar = generate_sequences(N_SEQS, seed=SEED)
    H = rollout_h(ec, cb, stim, ctx, Ostar, rstar, device)     # (N,T,256)
    stim_n, ctx_n = stim.numpy(), ctx.numpy()
    tss = trials_since_switch(ctx_n)
    Hflat = H.reshape(-1, H.shape[-1])

    for shift_prev, tag, title in [
        (False, "fig5_poststim_MTL.png",
         "MTL (h_t) — Fig-5 analog: CURRENT-trial variables (post-stimulus)"),
        (True,  "fig3_prestim_MTL.png",
         "MTL (h_t) — Fig-3 analog: PREVIOUS-trial variables (pre-stimulus)"),
    ]:
        print(f"\n[{tag}]", flush=True)
        cond2d, valid2d = cond_labels(stim_n, ctx_n, shift_prev=shift_prev)
        cond = cond2d.reshape(-1); valid = valid2d.reshape(-1)
        res = analyze_pooled(Hflat[valid], cond[valid], rng)
        tss_res = over_tss(Hflat, cond, valid2d, tss, rng)
        named = {nm: (res["ccgp"][i], res["ps"][i]) for i, nm in res["name_of"].items()}
        print(f"   SD={res['sd']:.3f} | " +
              " ".join(f"{nm}: CCGP={c:.2f} PS={p:.2f}" for nm, (c, p) in named.items()))
        plot_panels(res, tss_res, title, tag)


if __name__ == "__main__":
    main()

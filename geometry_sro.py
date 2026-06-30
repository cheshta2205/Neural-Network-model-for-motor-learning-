"""
geometry_sro.py — Bernardi et al. 2020 (Cell) Fig-3/Fig-5 geometry analysis on the
SRO context-inference model (context_inference_sro_rpe.py).

For the 8 task conditions (4 stim x 2 ctx) we compute, over all 35 balanced
dichotomies, the four Bernardi measures on three model "populations":

  h_t   : EC GRU hidden state (256-D)   -> MTL / HPC analog          [PRIMARY]
  z_ec  : EC readout latent  (32-D)     -> bottleneck context code   [contrast]
  cb_h  : CB trunk activity  (256-D)    -> downstream / "PFC-like"    [contrast]
- also look at geometry_sro_panels.py

Run:  python geometry_sro.py            (uses the saved checkpoint)
"""

import os
from itertools import combinations, permutations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

from context_inference_sro_rpe import (
    EntorhinalCortex, CerebellumCB, generate_sequences, load,
    SRO, N_STIM, N_CTX,
)

# ---- config
N_SEQS        = 300           # rollout sequences (each SEQ_LEN trials)
SEED          = 999
TRIALS_PER_COND = 60          # subsample per condition (speed)
CCGP_TRAIN_PER_SIDE = 2       # train on 2+2 conditions, test on held-out 2+2
N_NULL        = 20            # shuffles for the chance baseline
OUT = "Paper/neural_activity_comparisons/sro/geometry_sro.png"

COLORS = {"context": "#d62728", "value": "#e377c2", "action": "#ffba08"}
REPS_ORDER = ["h_t (MTL)", "z_ec", "cb_h (PFC-like)"]


# ---- rollout that logs internal populations
def rollout_internals(ec, cb, stim, ctx, Ostar, rstar, device):
    """Return per-trial (h_post, z_ec, cb_h) plus labels, all flattened to (n_trials, D)."""
    B, T = stim.shape
    x_oh = F.one_hot(stim, N_STIM).float().to(device)
    rstar_d = rstar.to(device)

    h = ec.init_hidden(B, device)
    x_prev = torch.zeros(B, N_STIM, device=device)
    delta_pr = torch.zeros(B, 1, device=device)

    H, Z, CBH = [], [], []
    with torch.no_grad():
        for t in range(T):
            x_t = x_oh[:, t]
            ec_in = torch.cat([x_prev, x_t, delta_pr], dim=-1)
            h_new = ec.cell(ec_in, h)
            z_t = ec.readout(h_new)
            cb_h = cb.trunk(torch.cat([x_t, z_t], dim=-1))
            r_pred = cb.reward_head(cb_h)

            H.append(h_new); Z.append(z_t); CBH.append(cb_h)   # h_t = post-stimulus (current trial)
            delta_pr = (rstar_d[:, t:t + 1] - r_pred).detach()
            x_prev = x_t
            h = h_new

    stack = lambda L: torch.stack(L, 1).cpu().numpy().reshape(B * T, -1)
    return {"h_t (MTL)": stack(H), "z_ec": stack(Z), "cb_h (PFC-like)": stack(CBH)}


# ---- conditions, variables, dichotomies
def condition_table():
    """cond id = ctx*4 + stim. Return per-condition (ctx, stim, operant, reward)."""
    ctx = np.array([c // 4 for c in range(8)])
    stim = np.array([c % 4 for c in range(8)])
    op = SRO[ctx, stim, 0].astype(int)
    rew = SRO[ctx, stim, 1].astype(int)
    return ctx, stim, op, rew


def all_dichotomies():
    items = list(range(8))
    seen, dichs = set(), []
    for combo in combinations(items, 4):
        P = frozenset(combo)
        if P in seen:
            continue
        N = frozenset(items) - P
        seen.add(P); seen.add(N)
        dichs.append((sorted(P), sorted(N)))
    return dichs                       # 35


def named_dichotomies():
    ctx, stim, op, rew = condition_table()
    def split(v):
        P = [c for c in range(8) if v[c] == 1]
        N = [c for c in range(8) if v[c] == 0]
        return (sorted(P), sorted(N))
    return {"context": split(ctx), "value": split(rew), "action": split(op)}


def cond_of_trial(stim, ctx):
    return (ctx * 4 + stim).reshape(-1)


# ---- the four measures
def subsample(X, cond, m=TRIALS_PER_COND, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = []
    for c in range(8):
        ci = np.where(cond == c)[0]
        if len(ci) > m:
            ci = rng.choice(ci, m, replace=False)
        idx.append(ci)
    idx = np.concatenate(idx)
    return X[idx], cond[idx]


def decode_accuracy(X, cond, P):
    y = np.isin(cond, P).astype(int)
    clf = LogisticRegression(max_iter=200)
    # SHUFFLE: data is ordered by condition, so unshuffled stratified folds hold out
    # whole conditions -> "decoding" silently becomes CCGP-like. Shuffle => true
    # pooled (same-condition train+test) decode, the way Bernardi's decoding is defined.
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    return cross_val_score(clf, X, y, cv=cv).mean()


def ccgp(X, cond, P, N, k=CCGP_TRAIN_PER_SIDE):
    accs = []
    for trP in combinations(P, k):
        teP = [c for c in P if c not in trP]
        for trN in combinations(N, k):
            teN = [c for c in N if c not in trN]
            tr = np.isin(cond, list(trP) + list(trN))
            te = np.isin(cond, teP + teN)
            ytr = np.isin(cond[tr], trP).astype(int)
            yte = np.isin(cond[te], teP).astype(int)
            clf = LogisticRegression(max_iter=200).fit(X[tr], ytr)
            accs.append((clf.predict(X[te]) == yte).mean())
    return float(np.mean(accs))


def parallelism(cond_means, P, N):
    best = -1.0
    for perm in permutations(N):
        vecs = []
        for i, p in enumerate(P):
            v = cond_means[p] - cond_means[perm[i]]
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
        cs = [vecs[i] @ vecs[j] for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        best = max(best, float(np.mean(cs)))
    return best


def analyze_rep(X, cond, rng):
    """Return dict with per-dichotomy decoding & CCGP & PS, named-variable values, SD, null."""
    Xs, cs = subsample(X, cond, rng=rng)
    cond_means = {c: Xs[cs == c].mean(0) for c in range(8)}
    dichs = all_dichotomies()
    named = named_dichotomies()
    # map named -> index in dichs (match as set, either orientation)
    name_of = {}
    for nm, (P, N) in named.items():
        sP = frozenset(P)
        for i, (dP, dN) in enumerate(dichs):
            if frozenset(dP) == sP or frozenset(dN) == sP:
                name_of[i] = nm
    dec, cc, ps = [], [], []
    for (P, N) in dichs:
        dec.append(decode_accuracy(Xs, cs, P))
        cc.append(ccgp(Xs, cs, P, N))
        ps.append(parallelism(cond_means, P, N))
    dec, cc, ps = map(np.array, (dec, cc, ps))
    sd = dec.mean()
    # shuffle null for SD
    null_sd = []
    for _ in range(N_NULL):
        perm = rng.permutation(cs)
        null_sd.append(np.mean([decode_accuracy(Xs, perm, P) for (P, N) in dichs[:10]]))
    return {"dec": dec, "ccgp": cc, "ps": ps, "sd": sd,
            "null_sd": np.array(null_sd), "name_of": name_of}


# ---- plotting (Bernardi colors/layout)
def plot(results):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    reps = REPS_ORDER
    x = np.arange(len(reps))

    # Panel A: decoding (open) + CCGP (filled gray), highlight context/value/action
    ax = axes[0]
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    for xi, rep in enumerate(reps):
        r = results[rep]
        n = len(r["ccgp"])
        jit = (np.random.default_rng(xi).random(n) - 0.5) * 0.28
        ax.scatter(xi + jit - 0.12, r["dec"], facecolors="none",
                   edgecolors="0.5", s=22, lw=0.8, label="decoding" if xi == 0 else None)
        ax.scatter(xi + jit + 0.12, r["ccgp"], color="0.55", s=18,
                   label="CCGP" if xi == 0 else None)
        for i, nm in r["name_of"].items():
            ax.scatter(xi + 0.12, r["ccgp"][i], color=COLORS[nm], s=80,
                       edgecolors="k", lw=0.6, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(reps, rotation=12)
    ax.set_ylabel("decoding acc. (open) / CCGP (filled)")
    ax.set_title("A — Decoding & CCGP over 35 dichotomies")
    loA = min(min(results[r]["dec"].min(), results[r]["ccgp"].min()) for r in reps)
    ax.set_ylim(loA - 0.05, 1.02); ax.legend(loc="lower left", fontsize=8)

    # Panel B: parallelism score
    ax = axes[1]
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    for xi, rep in enumerate(reps):
        r = results[rep]
        n = len(r["ps"])
        jit = (np.random.default_rng(xi + 9).random(n) - 0.5) * 0.28
        ax.scatter(xi + jit, r["ps"], color="0.6", s=18)
        for i, nm in r["name_of"].items():
            ax.scatter(xi, r["ps"][i], color=COLORS[nm], s=80, edgecolors="k", lw=0.6, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(reps, rotation=12)
    ax.set_ylabel("parallelism score")
    ax.set_title("B — Parallelism score")

    # Panel E: shattering dimensionality vs shuffle null
    ax = axes[2]
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    for xi, rep in enumerate(reps):
        r = results[rep]
        ax.bar(xi, r["sd"], width=0.5, color="0.7", edgecolor="k")
        ax.scatter([xi] * len(r["null_sd"]), r["null_sd"], color="crimson", s=10,
                   zorder=5, label="shuffle null" if xi == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(reps, rotation=12)
    ax.set_ylabel("shattering dimensionality")
    ax.set_title("E — SD (bar) vs shuffle null")
    loE = min(min(results[r]["sd"], results[r]["null_sd"].min()) for r in reps)
    ax.set_ylim(loE - 0.05, 1.02); ax.legend(fontsize=8)

    # legend for the three variables
    handles = [plt.Line2D([0], [0], marker="o", ls="", mec="k",
               mfc=COLORS[k], ms=9, label=k) for k in COLORS]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {OUT}")


def main():
    rng = np.random.default_rng(SEED)
    ec, cb, device = load()
    stim, ctx, Ostar, rstar = generate_sequences(N_SEQS, seed=SEED)
    reps = rollout_internals(ec, cb, stim, ctx, Ostar, rstar, device)
    cond = cond_of_trial(stim.numpy(), ctx.numpy())

    results = {}
    print("Computing geometry (decoding / CCGP / SD / PS) ...")
    for rep in REPS_ORDER:
        print(f"  {rep:18s} dim={reps[rep].shape[1]}", flush=True)
        results[rep] = analyze_rep(reps[rep], cond, rng)
        r = results[rep]
        named = {nm: (r["ccgp"][i], r["ps"][i]) for i, nm in r["name_of"].items()}
        print(f"     SD={r['sd']:.3f} | " +
              " ".join(f"{nm}: CCGP={c:.2f} PS={p:.2f}" for nm, (c, p) in named.items()))
    plot(results)


if __name__ == "__main__":
    main()

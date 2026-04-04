"""
plot_angle_overlay.py

Plots A vs A2 first_angle overlay for all available conditions in seed_0,
comparing MC+EC+CB (no HC) vs MC+EC+CB+HC.
Run this while mc_replay_hc.py is still running — reads whatever cond_*.npz exist.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SEED       = 0
N_PHASE_EP = 2000
SW         = 30
EP_SHOW    = 400
A_IDX      = 0
A2_IDX     = 2

DATA_DIR = f"/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/MC_EC_CB/MC_savings_norouting/replay_hc/seed_{SEED}"
FIG_DIR  = DATA_DIR

COND_NAMES = [
    "MC only",
    "MC + CB",
    "MC + EC",
    "MC + EC + CB",
    "MC + EC + CB + HC",
    "EC lesion (HC active)",
    "EC+HC lesion",
]
COND_COLORS = ["dimgray", "steelblue", "seagreen", "tomato", "darkorange", "salmon", "lightcoral"]
PHASE_LABELS = ["A: rot+60°", "B: curl+", "A2: rot+60°"]


def smooth(x, w=SW):
    x = np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x


def split_phases(arr):
    arr = np.array(arr)
    return [arr[i * N_PHASE_EP:(i + 1) * N_PHASE_EP] for i in range(3)]


ep_local = np.arange(1, N_PHASE_EP + 1)

# ── Load available conditions ─────────────────────────────────────────────────
available = []
for cond_idx, name in enumerate(COND_NAMES):
    path = os.path.join(DATA_DIR, f"cond_{cond_idx}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        curve = d["curve"].item()
        available.append((cond_idx, name, curve))
        print(f"Loaded: {name}")
    else:
        print(f"Missing: {name} (cond_{cond_idx})")

print(f"\n{len(available)} conditions loaded.")

if len(available) == 0:
    print("No data yet.")
    exit()

# ── Fig 1: Overlay — all available conditions, A vs A2 ───────────────────────
fig, axes = plt.subplots(len(available), 1,
                         figsize=(7, 3.5 * len(available)), squeeze=False)

for row, (cond_idx, name, curve) in enumerate(available):
    ax = axes[row, 0]
    color = COND_COLORS[cond_idx]
    phases = split_phases(curve["first_angle"])

    for ph_idx, col, lbl in [
        (A_IDX,  "steelblue",  "A (1st)"),
        (A2_IDX, "darkorange", "A2 (2nd)"),
    ]:
        if len(phases[ph_idx]) < SW:
            continue
        y_sm = smooth(np.array(phases[ph_idx]))
        ep_sm = ep_local[SW - 1:]
        mask  = ep_sm <= EP_SHOW
        ax.plot(ep_sm[mask], y_sm[mask], color=col, lw=2, label=lbl)

    ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° target")
    ax.set_xlim(1, EP_SHOW)
    ax.set_ylim(-110, 60)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylabel(f"{name}\nFirst angle (°)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if row == 0:
        ax.set_title(f"A vs A2 — first-step angle (seed {SEED})", fontsize=11)
    if row == len(available) - 1:
        ax.set_xlabel("Episode within phase", fontsize=9)

plt.tight_layout()
out = os.path.join(FIG_DIR, "angle_overlay_first.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# ── Fig 2: Single comparison panel — MC+EC+CB vs MC+EC+CB+HC ─────────────────
target_names = {"MC + EC + CB", "MC + EC + CB + HC"}
comp = [(idx, name, curve) for idx, name, curve in available if name in target_names]

if len(comp) >= 1:
    fig, ax = plt.subplots(figsize=(8, 4))
    styles = {
        "MC + EC + CB":      {"ls": "--", "lw": 2.0, "alpha": 0.85},
        "MC + EC + CB + HC": {"ls": "-",  "lw": 2.5, "alpha": 1.0},
    }
    for cond_idx, name, curve in comp:
        color = COND_COLORS[cond_idx]
        phases = split_phases(curve["first_angle"])
        kw = styles.get(name, {"ls": "-", "lw": 1.5, "alpha": 0.8})
        for ph_idx, col, lbl in [
            (A_IDX,  "steelblue",  "A (1st)"),
            (A2_IDX, "darkorange", "A2 (2nd)"),
        ]:
            if len(phases[ph_idx]) < SW:
                continue
            y_sm = smooth(np.array(phases[ph_idx]))
            ep_sm = ep_local[SW - 1:]
            mask  = ep_sm <= EP_SHOW
            ax.plot(ep_sm[mask], y_sm[mask],
                    color=col, label=f"{name} — {lbl}", **kw)

    ax.axhline(-60, color="gray", ls=":", lw=1.2, label="-60° target")
    ax.set_xlim(1, EP_SHOW)
    ax.set_ylim(-110, 60)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.set_xlabel("Episode within phase", fontsize=9)
    ax.set_ylabel("First-step angle (°)", fontsize=9)
    ax.set_title(f"Savings: no-HC vs replay-HC (seed {SEED})", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out2 = os.path.join(FIG_DIR, "angle_overlay_comparison.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")
else:
    print("Need MC+EC+CB and/or MC+EC+CB+HC to make comparison panel.")

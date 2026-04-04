"""
plot_hc_ppo_seeds.py

Aggregate all seed runs for MC_norouting_savings_HC_PPO and produce figures
with mean ± SEM across seeds.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Constants (must match MC_norouting_savings_HC_PPO.py) ─────────────────────
N_PHASE_EP  = 2000
SW          = 20

FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/MC_EC_CB/MC_savings_norouting/PPO/hc_ppo"
SEEDS   = [0, 1, 2, 3, 4, 5]

CONDITIONS = [
    "MC only",
    "MC + CB",
    "MC + EC",
    "MC + EC + CB",
    "MC + EC + CB + HC",
    "EC lesion (HC active)",
    "EC+HC lesion",
]

PHASE_LABELS = ["pre-B: curl+", "A: rot+60°", "B: curl+", "A2: rot+60°", "B2: curl+"]
A_IDX  = 1
B_IDX  = 2
A2_IDX = 3
PLOT_PHASES = [A_IDX, B_IDX, A2_IDX]

COND_COLORS = ["dimgray", "steelblue", "seagreen", "tomato", "darkorange", "salmon", "lightcoral"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def smooth(arr, w):
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def fit_asymptote(x, y):
    try:
        def model(t, L, y0, k): return L + (y0 - L) * np.exp(-k * t)
        p0 = [y[-1], y[0], 0.001]
        popt, _ = curve_fit(model, x, y, p0=p0, maxfev=4000)
        L, y0, k = popt
        if k <= 0: return None, None, None, None
        x95 = -np.log(0.05) / k
        return L, y0, k, x95
    except Exception:
        return None, None, None, None


def split_phases(arr):
    arr = np.array(arr)
    return [arr[i * N_PHASE_EP:(i + 1) * N_PHASE_EP] for i in range(len(PHASE_LABELS))]


# ── Load all seeds ────────────────────────────────────────────────────────────
# all_data[seed_idx] = list of dicts (one per condition), each with 'success', 'first_angle'
all_data = []
for seed in SEEDS:
    path = os.path.join(FIG_DIR, f"seed_{seed}", "hc_ppo_data.npz")
    d = np.load(path, allow_pickle=True)
    curves = list(d["curves"])   # list of dicts
    all_data.append(curves)

n_seeds = len(all_data)
n_conds = len(CONDITIONS)
ep_local = np.arange(1, N_PHASE_EP + 1)


# ── Build per-condition arrays across seeds ───────────────────────────────────
# success_all[cond][seed] = array (N_PHASE_EP * n_phases,)
# We smooth per seed then average.

def get_smoothed_phases(key, cond_idx):
    """Return list over seeds of list over phases of smoothed array."""
    out = []
    for seed_curves in all_data:
        curve = seed_curves[cond_idx]
        phases = split_phases(curve[key])
        smoothed = [smooth(np.array(p), SW) for p in phases]
        out.append(smoothed)
    return out  # [n_seeds][n_phases] each of length (N_PHASE_EP - SW + 1)


ep_sm = ep_local[SW - 1:]


def mean_sem(smoothed_phases, ph_idx):
    """Mean and SEM across seeds for a given phase index."""
    arr = np.stack([s[ph_idx] for s in smoothed_phases], axis=0)  # (n_seeds, T)
    return arr.mean(0), arr.std(0) / np.sqrt(n_seeds)


# ── Fig 1: Learning curves (A / B / A2) ──────────────────────────────────────
fig, axes = plt.subplots(n_conds, len(PLOT_PHASES), figsize=(15, 4 * n_conds), sharey=True)

for row, (cname, color) in enumerate(zip(CONDITIONS, COND_COLORS)):
    sp = get_smoothed_phases("success", row)
    for col, ph_idx in enumerate(PLOT_PHASES):
        ax = axes[row, col]
        mu, se = mean_sem(sp, ph_idx)
        ax.plot(ep_sm, mu, color=color, lw=2)
        ax.fill_between(ep_sm, mu - se, mu + se, color=color, alpha=0.25)

        # fit on mean
        L, y0, k, x95 = fit_asymptote(ep_sm, mu)
        if L is not None:
            ax.plot(ep_sm, L + (y0 - L) * np.exp(-k * ep_sm),
                    color="black", ls="--", lw=1, alpha=0.7)
            if x95 is not None and x95 < N_PHASE_EP:
                ax.axvline(x95, color="black", ls=":", lw=1)
                ax.text(x95, 0.05, f"ep{int(x95)}", fontsize=6, ha="center")

        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.set_title(PHASE_LABELS[ph_idx], fontsize=10)
        if col == 0:
            ax.set_ylabel(f"{cname}\nSuccess", fontsize=8)
        if row == n_conds - 1:
            ax.set_xlabel("Episode within phase", fontsize=8)

fig.suptitle(f"HC-PPO savings — success curves (A / B / A2)\nmean ± SEM, n={n_seeds} seeds", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hc_ppo_learning_curves_multiseed.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved hc_ppo_learning_curves_multiseed.png")


# ── Fig 2: A vs A2 first-angle overlay ───────────────────────────────────────
EP_SHOW = 400
fig, axes = plt.subplots(n_conds, 1, figsize=(6, 4 * n_conds), squeeze=False)

for row, (cname, color) in enumerate(zip(CONDITIONS, COND_COLORS)):
    ax = axes[row, 0]
    sp = get_smoothed_phases("first_angle", row)
    for ph_idx, alpha, lbl in [
        (A_IDX,  0.35, PHASE_LABELS[A_IDX] + " (1st)"),
        (A2_IDX, 1.00, PHASE_LABELS[A2_IDX] + " (2nd)"),
    ]:
        mu, se = mean_sem(sp, ph_idx)
        mask = ep_sm <= EP_SHOW
        ax.plot(ep_sm[mask], mu[mask], color=color, lw=2, alpha=alpha, label=lbl)
        ax.fill_between(ep_sm[mask], (mu - se)[mask], (mu + se)[mask],
                        color=color, alpha=alpha * 0.25)

    ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° target")
    ax.set_xlim(1, EP_SHOW)
    ax.set_ylim(-110, 60)
    ax.legend(fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    if row == 0:
        ax.set_title("A vs A2 — first-step angle", fontsize=10)
    ax.set_ylabel(f"{cname}\nAngle (°)", fontsize=8)
    if row == n_conds - 1:
        ax.set_xlabel("Episode within phase", fontsize=8)

fig.suptitle(f"HC-PPO: A vs A2 angle overlay (faded=1st, solid=2nd)\nmean ± SEM, n={n_seeds} seeds", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hc_ppo_angle_overlay_multiseed.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved hc_ppo_angle_overlay_multiseed.png")


# ── Fig 3: Savings bar (episodes to 95% asymptote, normalised) ───────────────
# Per seed: get x95 for A and A2, then average across seeds.
x95_A_all  = np.zeros((n_seeds, n_conds))
x95_A2_all = np.zeros((n_seeds, n_conds))

for si, seed_curves in enumerate(all_data):
    for ci in range(n_conds):
        sp = get_smoothed_phases("success", ci)
        for ph_idx, store in [(A_IDX, x95_A_all), (A2_IDX, x95_A2_all)]:
            y_sm = sp[si][ph_idx]
            _, _, _, x95 = fit_asymptote(ep_sm, y_sm)
            store[si, ci] = x95 if (x95 is not None and x95 < N_PHASE_EP) else N_PHASE_EP

norm_all = x95_A2_all / np.where(x95_A_all > 0, x95_A_all, 1.0)
norm_mu  = norm_all.mean(0)
norm_se  = norm_all.std(0) / np.sqrt(n_seeds)

fig, ax = plt.subplots(figsize=(8, 4))
width = 0.35
xs = np.arange(n_conds)

for i, (cname, color) in enumerate(zip(CONDITIONS, COND_COLORS)):
    ax.bar(i - width / 2, 1.0,        width, color=color, alpha=0.4)
    ax.bar(i + width / 2, norm_mu[i], width, color=color, alpha=1.0,
           yerr=norm_se[i], capsize=4, error_kw=dict(lw=1.5))
    ax.annotate("", xy=(i + width / 2, norm_mu[i]), xytext=(i + width / 2, 1.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(i + width / 2 + 0.08, (1.0 + norm_mu[i]) / 2,
            f"{norm_mu[i]:.2f}", fontsize=8, va="center")

ax.set_xticks(xs)
ax.set_xticklabels(CONDITIONS, fontsize=8, rotation=20, ha="right")
ax.set_ylabel("Episodes to 95% asymptote\n(normalised to A = 1.0)", fontsize=10)
ax.set_title(f"HC-PPO Savings: A vs A2\nmean ± SEM, n={n_seeds} seeds", fontsize=10)
ax.set_ylim(0, 1.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="A (1st)"),
                   Patch(facecolor="gray", alpha=1.0, label="A2 (2nd)")], fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hc_ppo_savings_bar_multiseed.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved hc_ppo_savings_bar_multiseed.png")


# ── Fig 4: EC lesion comparison ───────────────────────────────────────────────
name_map   = {c: i for i, c in enumerate(CONDITIONS)}
intact_key = "MC + EC + CB + HC"
lesion_pairs = [
    ("EC lesion (HC active)", "EC lesion only\n(HC gets wrong key)"),
    ("EC+HC lesion",          "EC+HC lesion\n(HC disabled)"),
]

if intact_key in name_map:
    ii  = name_map[intact_key]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, ph_idx, ph_lbl in [(axes[0], A_IDX, PHASE_LABELS[A_IDX]),
                                (axes[1], A2_IDX, PHASE_LABELS[A2_IDX])]:
        sp = get_smoothed_phases("first_angle", ii)
        mu, se = mean_sem(sp, ph_idx)
        mask = ep_sm <= 400
        ax.plot(ep_sm[mask], mu[mask], color=COND_COLORS[ii], lw=2, ls="-",
                label="EC intact (HC active)")
        ax.fill_between(ep_sm[mask], (mu - se)[mask], (mu + se)[mask],
                        color=COND_COLORS[ii], alpha=0.2)

        for (lkey, llabel), ls, alpha in zip(lesion_pairs, ["--", ":"], [0.8, 0.8]):
            if lkey in name_map:
                li = name_map[lkey]
                sp_l = get_smoothed_phases("first_angle", li)
                mu_l, se_l = mean_sem(sp_l, ph_idx)
                ax.plot(ep_sm[mask], mu_l[mask], color=COND_COLORS[li],
                        lw=2, alpha=alpha, ls=ls, label=llabel)
                ax.fill_between(ep_sm[mask], (mu_l - se_l)[mask], (mu_l + se_l)[mask],
                                color=COND_COLORS[li], alpha=0.15)

        ax.axhline(-60, color="gray", ls=":", lw=1.2, label="-60° target")
        ax.set_xlim(1, 400)
        ax.set_ylim(-110, 60)
        ax.set_title(ph_lbl, fontsize=10)
        ax.set_xlabel("Episode within phase", fontsize=9)
        ax.set_ylabel("First-step angle (°)", fontsize=9)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"EC lesion comparison — MC + EC + CB + HC\nmean ± SEM, n={n_seeds} seeds", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "hc_ppo_ec_lesion_multiseed.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved hc_ppo_ec_lesion_multiseed.png")

print("\nAll multi-seed figures saved.")

"""
MC_norouting_savings_HC_PPO_goal.py

PPO-style savings with HC + Action Buffer — HC conditioned on goal.

Mechanism
---------
- HC stores (key, a_ideal) at end of each phase.
  key = normalize(concat(normalize(h_ec_mean), goal_mean / GRID_SIZE))
  a_ideal = mean first-step action from last N successful episodes.
- At retrieval, query = normalize(concat(normalize(h_ec), goal / GRID_SIZE))
  so HC is sensitive to both context (h_ec) and the current goal.
- Motivation: on bad seeds, contaminated h_ec at phase transitions causes
  HC to retrieve the wrong a_ideal → PPO pulls actions the wrong way.
  Adding goal to the key gives an orthogonal signal that can break the tie.

Action buffer stores first-step actions from successful episodes (small, high-quality).
At each step:
    1. (h_ec, goal) → HC cosine sim → retrieve a_ideal for current context+goal
    2. From buffer, compute a* = weighted avg of buffer actions,
       weights = softmax(-β * ||a_i - a_ideal||²)
    3. Add PPO-style loss: λ * ||a_t - a*||²

Conditions
----------
1. MC only
2. MC + EC
3. MC + EC + HC
4. MC + EC + HC (EC lesion — h_ec zeroed → HC retrieval fails)

Protocol: BABAB (5 phases), plots show A / B / A2 (indices 1, 2, 3).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM  = 32
EC_HIDDEN   = 256
CB_HIDDEN   = 256
EC_INPUT    = 8
CB_PROJ_DIM = 32
MC_HIDDEN   = 256

MAX_STEPS   = 20
N_PHASE_EP  = 2000
GOAL_RADIUS = 0.1
GRID_SIZE   = 4.0
DT          = 1.0
LR_MC       = 1e-3

RESET_EC_PER_EPISODE = True

# HC / buffer
HC_N_IDEAL           = 200   # max individual memories stored per phase
BUFFER_CAPACITY      = 500   # max first-step actions from successful eps
HC_KL_WEIGHT         = 0.3   # λ: weight of PPO-style proximity loss
HC_RETRIEVAL_MAX_DIST = 0.5  # max combined L2 distance to activate HC
HC_GOAL_WEIGHT       = 1.0   # λ_g: weight of goal distance vs h_ec distance
SOFTMAX_BETA         = 5.0   # β: sharpness of action-divergence weighting

CKPT    = "checkpoints/ectocb_8ctx_seq80_z32.pt"
FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/MC_EC_CB/MC_savings_norouting/PPO/hc_ppo_goal"

# ── Contexts / phases ─────────────────────────────────────────────────────────
CTX_A = (1.0, np.pi / 3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B = (1.0, 0.0,      +0.4, 0.0, 0.0, 0.0, "B: curl+")

PHASES       = [CTX_B, CTX_A, CTX_B, CTX_A, CTX_B]
PHASE_LABELS = ["pre-B: curl+", "A: rot+60°", "B: curl+", "A2: rot+60°", "B2: curl+"]

A_IDX  = 1
B_IDX  = 2
A2_IDX = 3

# ── Conditions ────────────────────────────────────────────────────────────────
CONDITIONS = [
    # name                              use_ec  use_cb  retrain_cb_noec  use_hc  ec_lesion
    {"name": "MC only",                 "use_ec": False, "use_cb": False, "retrain_cb_noec": False, "use_hc": False, "ec_lesion": False},
    {"name": "MC + CB",                 "use_ec": False, "use_cb": True,  "retrain_cb_noec": True,  "use_hc": False, "ec_lesion": False},
    {"name": "MC + EC",                 "use_ec": True,  "use_cb": False, "retrain_cb_noec": False, "use_hc": False, "ec_lesion": False},
    {"name": "MC + EC + CB",            "use_ec": True,  "use_cb": True,  "retrain_cb_noec": False, "use_hc": False, "ec_lesion": False},
    {"name": "MC + EC + CB + HC",       "use_ec": True,  "use_cb": True,  "retrain_cb_noec": False, "use_hc": True,  "ec_lesion": False},
    # EC lesion only: h_ec=0 but HC still active → gets zero key → wrong a_ideal → PPO pulls wrong way
    {"name": "EC lesion (HC active)",   "use_ec": True,  "use_cb": True,  "retrain_cb_noec": False, "use_hc": True,  "ec_lesion": True},
    # EC+HC lesion: h_ec=0 AND HC disabled → same as MC+CB baseline
    {"name": "EC+HC lesion",            "use_ec": True,  "use_cb": True,  "retrain_cb_noec": False, "use_hc": False, "ec_lesion": True},
]

COND_COLORS = ["dimgray", "steelblue", "seagreen", "tomato", "darkorange", "salmon", "lightcoral"]
KEEP_IDX    = [0, 1, 2, 3, 4, 6]

# ── Plot helpers ──────────────────────────────────────────────────────────────
from scipy.optimize import curve_fit

SW     = 30
SW_ACC = 30
SW_SAV = 30
SW_RAW = 60
ep_local = np.arange(1, N_PHASE_EP + 1)

def smooth(x, w=SW):
    x = np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x

def exp_model(xv, L, y0, k):
    return L + (y0 - L) * np.exp(-k * xv)

def fit_asymptote(x, y):
    try:
        x, y = np.asarray(x, float), np.asarray(y, float)
        p0 = [float(y[-1]), float(y[0]), 1e-3]
        popt, _ = curve_fit(exp_model, x, y, p0=p0, maxfev=5000)
        L, y0, k = popt
        x95 = -np.log(0.05) / k if (k > 0 and abs(L - y0) > 1e-6) else None
        return L, y0, k, x95
    except Exception:
        return None, None, None, None

def split_phases(arr):
    arr = np.array(arr)
    return [arr[i * N_PHASE_EP:(i + 1) * N_PHASE_EP] for i in range(len(PHASES))]

from matplotlib.patches import Patch

def plot_seed(seed, curves_all, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    curves      = [curves_all[i] for i in KEEP_IDX]
    cond_names  = [CONDITIONS[i]["name"] for i in KEEP_IDX]
    cond_colors = [COND_COLORS[i] for i in KEEP_IDX]
    n_conds     = len(curves)
    intact_name   = "MC + EC + CB + HC"
    lesion_name   = "EC+HC lesion"
    name_to_curve = {CONDITIONS[i]["name"]: curves_all[i] for i in range(len(CONDITIONS))}
    name_to_color = {CONDITIONS[i]["name"]: COND_COLORS[i] for i in range(len(CONDITIONS))}
    EP_SHOW = 400

    # Fig 1: Learning curves
    fig, axes = plt.subplots(n_conds, 1, figsize=(12, 3 * n_conds), squeeze=False)
    phase_boundaries = [i * N_PHASE_EP for i in range(1, len(PHASES))]
    for row, (name, color, curve) in enumerate(zip(cond_names, cond_colors, curves)):
        ax = axes[row, 0]
        succ = np.array(curve["success"]); y_sm = smooth(succ)
        ep_g = np.arange(1, len(succ) + 1)
        ax.plot(ep_g[SW - 1:], y_sm, color=color, lw=1.5)
        for xb in phase_boundaries:
            ax.axvline(xb, color="gray", ls="--", lw=0.8)
        for i, lbl in enumerate(PHASE_LABELS):
            ax.text((i + 0.5) * N_PHASE_EP, 1.02, lbl, ha="center", fontsize=7,
                    transform=ax.get_xaxis_transform())
        ax.set_ylim(0, 1.1); ax.set_ylabel(name, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if row == n_conds - 1: ax.set_xlabel("Episode (global)", fontsize=9)
    fig.suptitle(f"PPO Goal HC — Learning Curves (seed {seed})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/ppo_goal_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2: A vs A2 angle overlay
    fig, axes = plt.subplots(n_conds, 1, figsize=(6, 4 * n_conds), squeeze=False)
    for row, (name, curve) in enumerate(zip(cond_names, curves)):
        ax = axes[row, 0]
        phases = split_phases(curve["angle"])
        for ph_idx, col, lbl in [
            (A_IDX,  "steelblue",  PHASE_LABELS[A_IDX] + " (1st)"),
            (A2_IDX, "darkorange", PHASE_LABELS[A2_IDX] + " (2nd)"),
        ]:
            y_sm = smooth(np.array(phases[ph_idx])); ep_sm = ep_local[SW - 1:]
            ax.plot(ep_sm[ep_sm <= EP_SHOW], y_sm[ep_sm <= EP_SHOW], color=col, lw=2, label=lbl)
        ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° target")
        ax.set_xlim(1, EP_SHOW); ax.set_ylim(-110, 60)
        ax.legend(fontsize=7, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
        if row == 0: ax.set_title("A vs A2 — mean trajectory angle", fontsize=10)
        ax.set_ylabel(f"{name}\nAngle (°)", fontsize=8)
        if row == n_conds - 1: ax.set_xlabel("Episode within phase", fontsize=8)
    fig.suptitle(f"PPO Goal HC: A vs A2 angle overlay (seed {seed})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/ppo_goal_angle_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3: Savings bar
    fig, ax = plt.subplots(figsize=(7, 4))
    x95_A, x95_A2 = [], []
    for curve in curves:
        phases = split_phases(curve["success"])
        for ph_idx, store in [(A_IDX, x95_A), (A2_IDX, x95_A2)]:
            y_sm = smooth(np.array(phases[ph_idx])); ep_sm = ep_local[SW - 1:]
            _, _, _, x95 = fit_asymptote(ep_sm, y_sm)
            store.append(x95 if (x95 is not None and x95 < N_PHASE_EP) else N_PHASE_EP)
    width = 0.35; xs = np.arange(n_conds)
    for i, (name, color, a, a2) in enumerate(zip(cond_names, cond_colors, x95_A, x95_A2)):
        norm_a2 = a2 / a if a > 0 else 1.0
        ax.bar(i - width / 2, 1.0, width, color=color, alpha=0.4)
        ax.bar(i + width / 2, norm_a2, width, color=color, alpha=1.0)
        ax.annotate("", xy=(i + width / 2, norm_a2), xytext=(i + width / 2, 1.0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.text(i + width / 2 + 0.08, (1.0 + norm_a2) / 2, f"{norm_a2:.2f}", fontsize=8, va="center")
    ax.set_xticks(xs); ax.set_xticklabels(cond_names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Episodes to 95% asymptote\n(normalised to A = 1.0)", fontsize=10)
    ax.set_title(f"PPO Goal HC Savings: A vs A2 (seed {seed})", fontsize=10); ax.set_ylim(0, 1.5)
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="A (1st)"),
                       Patch(facecolor="gray", alpha=1.0, label="A2 (2nd)")], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/ppo_goal_savings_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 4: EC lesion comparison
    if intact_name in name_to_curve and lesion_name in name_to_curve:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, ph_idx, ph_lbl in [(axes[0], A_IDX, PHASE_LABELS[A_IDX]),
                                    (axes[1], A2_IDX, PHASE_LABELS[A2_IDX])]:
            for name, ls, lw, alpha in [(intact_name, "-", 2.0, 1.0), (lesion_name, "--", 2.0, 0.8)]:
                phases = split_phases(name_to_curve[name]["first_angle"])
                y_sm = smooth(np.array(phases[ph_idx])); ep_sm = ep_local[SW - 1:]
                ax.plot(ep_sm[ep_sm <= 400], y_sm[ep_sm <= 400], color=name_to_color[name],
                        lw=lw, alpha=alpha, ls=ls, label=name)
            ax.axhline(-60, color="gray", ls=":", lw=1.2)
            ax.set_xlim(1, 400); ax.set_ylim(-110, 60); ax.set_title(ph_lbl, fontsize=10)
            ax.set_xlabel("Episode within phase", fontsize=9)
            ax.set_ylabel("Mean trajectory angle (°)", fontsize=9)
            ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"EC+HC lesion vs intact (seed {seed})", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/ppo_goal_ec_lesion.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Fig 4b: Overlay
    if intact_name in name_to_curve and lesion_name in name_to_curve:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, ph_idx, col, ls, lbl in [
            (intact_name, A_IDX,  "steelblue",  "-",  "Intact — A (1st)"),
            (intact_name, A2_IDX, "darkorange", "-",  "Intact — A2 (2nd)"),
            (lesion_name, A_IDX,  "steelblue",  "--", "Lesion — A (1st)"),
            (lesion_name, A2_IDX, "darkorange", "--", "Lesion — A2 (2nd)"),
        ]:
            phases = split_phases(name_to_curve[name]["first_angle"])
            y_sm = smooth(np.array(phases[ph_idx])); ep_sm = ep_local[SW - 1:]
            ax.plot(ep_sm[ep_sm <= EP_SHOW], y_sm[ep_sm <= EP_SHOW], color=col, ls=ls, lw=2, label=lbl)
        ax.axhline(-60, color="gray", ls=":", lw=1.2)
        ax.set_xlim(1, EP_SHOW); ax.set_ylim(-110, 60)
        ax.set_xlabel("Episode within phase", fontsize=9); ax.set_ylabel("Mean trajectory angle (°)", fontsize=9)
        ax.set_title(f"Intact vs EC+HC lesion — A & A2 (seed {seed})", fontsize=10)
        ax.legend(fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/ppo_goal_ec_lesion_overlay.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Fig 5: ABA accuracy
    if intact_name in name_to_curve and lesion_name in name_to_curve:
        fig, ax = plt.subplots(figsize=(9, 4))
        for name, ls, col in [(intact_name, "-", "steelblue"), (lesion_name, "--", "darkorange")]:
            succ = np.array(name_to_curve[name]["success"])
            aba_succ = succ[N_PHASE_EP:4 * N_PHASE_EP]
            y_sm = smooth(aba_succ, SW_ACC)
            ax.plot(np.arange(1, len(y_sm) + 1), y_sm, color=col, lw=2, ls=ls, label=name)
        ax.axvline(N_PHASE_EP - SW + 1, color="gray", ls="--", lw=0.8)
        ax.axvline(2 * N_PHASE_EP - SW + 1, color="gray", ls="--", lw=0.8)
        for i, lbl in enumerate(["A: rot+60°", "B: curl+", "A2: rot+60°"]):
            ax.text((i + 0.5) * N_PHASE_EP, 1.02, lbl, ha="center", fontsize=8,
                    transform=ax.get_xaxis_transform())
        ax.set_ylim(0, 1.1); ax.set_xlabel("Episode within ABA", fontsize=9)
        ax.set_ylabel("Success rate (smoothed)", fontsize=9)
        ax.set_title(f"ABA accuracy: intact vs EC+HC lesion (seed {seed})", fontsize=10)
        ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/ppo_goal_aba_accuracy_lesion.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Fig 6: Savings accuracy
    if intact_name in name_to_curve and lesion_name in name_to_curve:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, name, title in [(axes[0], intact_name, "Intact"), (axes[1], lesion_name, "EC+HC lesion")]:
            succ_phases = split_phases(name_to_curve[name]["success"])
            for ph_idx, col, lbl in [(A_IDX, "steelblue", "A (1st)"), (A2_IDX, "darkorange", "A2 (2nd)")]:
                y_raw = np.array(succ_phases[ph_idx])
                y_sm = smooth(y_raw, SW_SAV); ep_sm = ep_local[SW_SAV - 1:]
                ax.plot(ep_sm, y_sm, color=col, lw=1.5, label=lbl)
                L, y0, k, x95 = fit_asymptote(ep_sm, y_sm)
                if L is not None:
                    ax.plot(ep_sm, exp_model(ep_sm, L, y0, k), color=col, lw=2, ls="--", alpha=0.9)
                if x95 is not None and 0 < x95 < N_PHASE_EP:
                    ax.axvline(x95, color=col, lw=1.2, ls=":", alpha=0.8)
                    ax.text(x95 + 15, 0.08, f"{x95:.0f}", color=col, fontsize=7, va="bottom")
            ax.set_ylim(0, 1.1); ax.set_xlim(1, N_PHASE_EP)
            ax.set_xlabel("Episode within phase", fontsize=9); ax.set_ylabel("Success rate", fontsize=9)
            ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Savings: A vs A2 accuracy (seed {seed})", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/ppo_goal_savings_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Fig 6c: Raw fit
    if intact_name in name_to_curve and lesion_name in name_to_curve:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, name, title in [(axes[0], intact_name, "Intact"), (axes[1], lesion_name, "EC+HC lesion")]:
            succ_phases = split_phases(name_to_curve[name]["success"])
            for ph_idx, col, lbl in [(A_IDX, "steelblue", "A (1st)"), (A2_IDX, "darkorange", "A2 (2nd)")]:
                y_raw = np.array(succ_phases[ph_idx])
                y_sm = smooth(y_raw, SW_RAW); ep_sm = ep_local[SW_RAW - 1:]
                ax.plot(ep_sm, y_sm, color=col, lw=1.5, label=lbl)
                L, y0, k, x95 = fit_asymptote(ep_local, y_raw)
                if L is not None:
                    ax.plot(ep_local, exp_model(ep_local, L, y0, k), color=col, lw=2, ls="--", alpha=0.9)
                if x95 is not None and 0 < x95 < N_PHASE_EP:
                    ax.axvline(x95, color=col, lw=1.2, ls=":", alpha=0.8)
            ax.set_ylim(0, 1.1); ax.set_xlim(1, N_PHASE_EP)
            ax.set_xlabel("Episode within phase", fontsize=9); ax.legend(fontsize=8)
            ax.set_title(title, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Savings raw fit (seed {seed})", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/ppo_goal_savings_raw_fit.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Fig 6d: Angle slope
    if intact_name in name_to_curve and lesion_name in name_to_curve:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, name, title in [(axes[0], intact_name, "Intact"), (axes[1], lesion_name, "EC+HC lesion")]:
            angle_phases = split_phases(name_to_curve[name]["first_angle"])
            for ph_idx, col, lbl in [(A_IDX, "steelblue", "A (1st)"), (A2_IDX, "darkorange", "A2 (2nd)")]:
                y_sm = smooth(np.array(angle_phases[ph_idx]), 20); ep_sm = ep_local[20 - 1:]
                dy = np.diff(y_sm); dy_sm = smooth(dy, 20); ep_dy = ep_sm[:len(dy_sm)]
                ax.plot(ep_dy[ep_dy <= 150], dy_sm[ep_dy <= 150], color=col, lw=1.5, label=lbl)
            ax.axhline(0, color="gray", ls="--", lw=0.8); ax.set_xlim(1, 150)
            ax.set_xlabel("Episode within phase", fontsize=9)
            ax.set_ylabel("d(angle)/d(episode)", fontsize=9)
            ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Slope of angle descent (seed {seed})", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/ppo_goal_angle_slope.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Fig 6b: Adaptation rate k
    if intact_name in name_to_curve and lesion_name in name_to_curve:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, name, title in [(axes[0], intact_name, "Intact"), (axes[1], lesion_name, "EC+HC lesion")]:
            succ_phases = split_phases(name_to_curve[name]["success"])
            ks, lbls, cols = [], [], []
            for ph_idx, col, lbl in [(A_IDX, "steelblue", "A (1st)"), (A2_IDX, "darkorange", "A2 (2nd)")]:
                y_raw = np.array(succ_phases[ph_idx])
                y_sm = smooth(y_raw, SW_SAV); ep_sm = ep_local[SW_SAV - 1:]
                L, y0, k, _ = fit_asymptote(ep_sm, y_sm)
                ks.append(k if k is not None else 0); lbls.append(lbl); cols.append(col)
            xs = np.arange(len(ks))
            ax.bar(xs, ks, color=cols, width=0.5, alpha=0.85)
            ax.set_xticks(xs); ax.set_xticklabels(lbls, fontsize=9)
            for x, k_val in zip(xs, ks):
                ax.text(x, k_val + 0.0001, f"{k_val:.4f}", ha="center", fontsize=8)
            ax.set_ylabel("Adaptation rate (k)", fontsize=9); ax.set_title(title, fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Adaptation rate k — A vs A2 (seed {seed})", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/ppo_goal_savings_slope.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Fig 7: ABA grid
    ABA_PHASE_INDICES = [A_IDX, B_IDX, A2_IDX]
    ABA_PHASE_LABELS  = [PHASE_LABELS[i] for i in ABA_PHASE_INDICES]
    fig, axes = plt.subplots(n_conds, 3, figsize=(12, 3 * n_conds), squeeze=False)
    for row, (name, color, curve) in enumerate(zip(cond_names, cond_colors, curves)):
        succ_phases = split_phases(curve["success"])
        for col, ph_idx in enumerate(ABA_PHASE_INDICES):
            ax = axes[row, col]
            y_raw = np.array(succ_phases[ph_idx])
            y_sm = smooth(y_raw, SW_ACC); ep_sm = ep_local[SW_ACC - 1:]
            ax.plot(ep_sm, y_sm, color=color, lw=1.8)
            y_sm_fit = smooth(y_raw, SW); ep_sm_fit = ep_local[SW - 1:]
            L, y0, k, x95 = fit_asymptote(ep_sm_fit, y_sm_fit)
            if L is not None:
                ax.plot(ep_sm_fit, exp_model(ep_sm_fit, L, y0, k), color="black", lw=1.2, ls="--", alpha=0.7)
                if x95 is not None and x95 < N_PHASE_EP:
                    ax.axvline(x95, color="black", lw=1, ls=":", alpha=0.6)
                    ax.text(x95 + 20, 0.1, f"x95={x95:.0f}", fontsize=6, color="black")
            ax.set_ylim(0, 1.1); ax.set_xlim(ep_sm[0], ep_sm[-1])
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0: ax.set_title(ABA_PHASE_LABELS[col], fontsize=9)
            if col == 0: ax.set_ylabel(f"{name}\nSuccess", fontsize=7)
            if row == n_conds - 1: ax.set_xlabel("Episode within phase", fontsize=8)
    fig.suptitle(f"ABA accuracy + asymptote fit — all conditions (seed {seed})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/ppo_goal_aba_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Seed {seed}: all 11 figures saved to {fig_dir}")


# ── Action Buffer ─────────────────────────────────────────────────────────────
class ActionBuffer:
    """
    Stores first-step actions from successful episodes.
    Small (≤ BUFFER_CAPACITY) and high-quality (success-filtered).

    Retrieval: weighted avg of stored actions,
    weights = softmax(-β * ||a_i - a_ideal||²).
    Actions that diverge least from HC's ideal get highest weight.
    """
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.capacity = capacity
        self._buf = deque(maxlen=capacity)   # each entry: tensor (2,) cpu

    def push(self, a):
        self._buf.append(a.detach().cpu())

    def __len__(self):
        return len(self._buf)

    def retrieve(self, a_ideal, device):
        """
        a_ideal: tensor (2,) on device.
        Returns a* = softmax-weighted average of buffer actions.
        """
        if len(self._buf) == 0:
            return a_ideal

        a_stack  = torch.stack(list(self._buf)).to(device)   # (N, 2)
        a_id     = a_ideal.detach()
        dists    = torch.sum((a_stack - a_id.unsqueeze(0)) ** 2, dim=1)   # (N,)
        weights  = torch.softmax(-SOFTMAX_BETA * dists, dim=0)            # (N,)
        a_star   = (weights.unsqueeze(1) * a_stack).sum(0)                # (2,)
        return a_star


# ── Hippocampus ───────────────────────────────────────────────────────────────
class Hippocampus:
    """
    Stores individual (h_ec, goal, a) tuples from successful episodes.
    Up to HC_N_IDEAL memories per phase are kept (the last N successful eps).

    Retrieval: combined L2 distance over (h_ec, goal) space.
      d_i = ||h_ec_norm - h_ec_i_norm||²  +  HC_GOAL_WEIGHT * ||g/4 - g_i/4||²
    Pick argmin(d_i) → return that episode's a as a_ideal.
    No averaging: each stored memory is one real episode, so goal info is preserved.
    """
    def __init__(self):
        self.h_ec_mem = []   # list of (EC_HIDDEN,) unit-norm tensors, cpu
        self.goal_mem = []   # list of (2,) tensors scaled to [0,1], cpu
        self.a_mem    = []   # list of (2,) tensors, cpu

    def write(self, h_ec_list, a_list, label, goal_list):
        """
        Stores each (h_ec, goal, a) individually — no averaging.
        h_ec_list : list of (1, EC_HIDDEN) tensors
        a_list    : list of (2,) tensors (first-step actions)
        goal_list : list of (2,) tensors (goal positions)
        """
        for h, a, g in zip(h_ec_list, a_list, goal_list):
            self.h_ec_mem.append(F.normalize(h.squeeze(0).detach().cpu(), dim=0))
            self.goal_mem.append(g.detach().cpu() / GRID_SIZE)
            self.a_mem.append(a.detach().cpu())

    def retrieve(self, h_ec, goal, device):
        """
        h_ec: (1, EC_HIDDEN) tensor.
        goal: (1, 2) or (2,) tensor — current episode goal position.
        Returns (a_ideal tensor on device, min_dist float).

        Combined L2:
          d_i = ||h_ec_norm - h_ec_i||²  +  HC_GOAL_WEIGHT * ||g/4 - g_i||²
        Activate HC only if min_dist <= HC_RETRIEVAL_MAX_DIST.
        """
        if not self.h_ec_mem:
            return None, float("inf")

        h_norm  = F.normalize(h_ec.squeeze(0).detach().cpu(), dim=0)
        g_query = goal.squeeze(0).detach().cpu() / GRID_SIZE

        h_stack = torch.stack(self.h_ec_mem)                          # (N, EC_HIDDEN)
        g_stack = torch.stack(self.goal_mem)                          # (N, 2)

        d_hec  = torch.sum((h_stack - h_norm.unsqueeze(0)) ** 2, dim=1)   # (N,)
        d_goal = torch.sum((g_stack - g_query.unsqueeze(0)) ** 2, dim=1)  # (N,)
        dist   = d_hec + HC_GOAL_WEIGHT * d_goal                          # (N,)

        best     = int(dist.argmin())
        min_dist = float(dist[best])
        return self.a_mem[best].to(device), min_dist


# ── Models ────────────────────────────────────────────────────────────────────
class EntorhinalCortex(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = EC_HIDDEN
        self.cell    = nn.GRUCell(EC_INPUT, EC_HIDDEN)
        self.readout = nn.Sequential(nn.Linear(EC_HIDDEN, LATENT_DIM), nn.Tanh())

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, EC_HIDDEN, device=device)

    def step(self, x, h):
        h_new = self.cell(x, h)
        return self.readout(h_new), h_new


class CerebellumCB(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(4 + LATENT_DIM, CB_HIDDEN), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(CB_HIDDEN, CB_HIDDEN), nn.ReLU())
        self.out    = nn.Linear(CB_HIDDEN, 2)

    def forward(self, s, a, z):
        h = self.layer1(torch.cat([s, a, z], dim=-1))
        return self.out(self.layer2(h))

    def hidden(self, s, a, z):
        return self.layer2(self.layer1(torch.cat([s, a, z], dim=-1)))


class CerebellumCB_NoEC(nn.Module):
    """CB retrained without EC — used in MC+CB condition."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(4, CB_HIDDEN), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(CB_HIDDEN, CB_HIDDEN), nn.ReLU())
        self.out    = nn.Linear(CB_HIDDEN, 2)

    def forward(self, s, a):
        h = self.layer1(torch.cat([s, a], dim=-1))
        return self.out(self.layer2(h))

    def hidden(self, s, a):
        return self.layer2(self.layer1(torch.cat([s, a], dim=-1)))


class MotorCortex(nn.Module):
    def __init__(self, use_ec=True, use_cb=False):
        super().__init__()
        self.use_ec = use_ec
        self.use_cb = use_cb
        input_dim   = 2 + 2   # state + goal
        if use_ec:
            input_dim += LATENT_DIM
            self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        if use_cb:
            input_dim += CB_PROJ_DIM
            W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
            self.register_buffer("cb_proj", W / W.norm(dim=0, keepdim=True))
        self.hidden_dim = MC_HIDDEN
        self.cell    = nn.GRUCell(input_dim, MC_HIDDEN)
        self.readout = nn.Linear(MC_HIDDEN, 2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, h_ec, g, h_mc, cb_hidden=None):
        parts = [s_t]
        if self.use_ec:
            z_ec = torch.tanh(self.ec_readout(h_ec))
            parts.append(z_ec)
        if self.use_cb and cb_hidden is not None:
            parts.append(cb_hidden @ self.cb_proj)
        parts.append(g)
        inp   = torch.cat(parts, dim=-1)
        h_new = self.cell(inp, h_mc)
        return torch.tanh(self.readout(h_new)), h_new


# ── Environment ───────────────────────────────────────────────────────────────
def env_step_torch(pos, vel, action, beta, theta, gamma, g, bx, by, device):
    cos_t = torch.tensor(float(np.cos(theta)), dtype=torch.float32, device=device)
    sin_t = torch.tensor(float(np.sin(theta)), dtype=torch.float32, device=device)
    R    = torch.stack([torch.stack([cos_t, -sin_t]),
                        torch.stack([sin_t,  cos_t])])
    curl = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32, device=device)
    b_v  = torch.tensor([float(bx), float(by)], dtype=torch.float32, device=device)
    F    = beta * (R @ action) + gamma * (curl @ vel) + g * b_v
    vel_new = vel + F
    pos_new = torch.clamp(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new.detach()


# ── Single episode ─────────────────────────────────────────────────────────────
def run_episode(
    ec, cb_frozen, mc, device, ctx, rng,
    train=True, use_ec=True, use_cb=False, cb_noec=None,
    hc=None, action_buffer=None, use_hc=False, ec_lesion=False,
):
    beta, theta, gamma, g, bx, by, _ = ctx

    start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(start - goal) < 0.5:
        goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    g_t  = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos  = torch.tensor(start, dtype=torch.float32, device=device)
    vel  = torch.zeros(2, dtype=torch.float32, device=device)

    h_ec   = ec.init_hidden(1, device)   # reset per episode
    h_mc   = mc.init_hidden(1, device)
    z_t    = torch.zeros(1, LATENT_DIM, device=device)
    prev_a = torch.zeros(1, 2, device=device)

    step_losses = []
    dist_seq    = []
    angle_seq   = []
    success     = False
    steps       = 0
    first_angle = None
    first_a     = None
    init_dist   = torch.norm(pos - g_t).item()

    for t in range(MAX_STEPS):
        d = torch.norm(pos - g_t).item()
        dist_seq.append(d)
        if d < GOAL_RADIUS:
            success = True
            break

        # ec_lesion: zero both h_ec into MC and z_t into CB
        z_for_cb = torch.zeros(1, LATENT_DIM, device=device) if ec_lesion else z_t
        with torch.no_grad() if not train else torch.enable_grad():
            if use_cb and cb_noec is not None:
                cb_h = cb_noec.hidden(pos.unsqueeze(0), prev_a)
            elif use_cb:
                cb_h = cb_frozen.hidden(pos.unsqueeze(0), prev_a, z_for_cb)
            else:
                cb_h = None

        h_ec_mc = torch.zeros(1, EC_HIDDEN, device=device) if (not use_ec or ec_lesion) else h_ec

        a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec_mc, g_t.unsqueeze(0), h_mc, cb_h)

        s_next, vel_next = env_step_torch(
            pos, vel, a_t.squeeze(0), beta, theta, gamma, g, bx, by, device
        )

        if train:
            step_losses.append(torch.sum((g_t - s_next) ** 2))
            if cb_noec is not None:
                pred = cb_noec(pos.unsqueeze(0), a_t.detach())
                step_losses.append(torch.sum((s_next.detach().unsqueeze(0) - pred) ** 2))

            # PPO-style HC loss: only when both HC and EC are genuinely active
            if use_hc and use_ec and not ec_lesion and hc is not None and action_buffer is not None:
                a_ideal, min_dist = hc.retrieve(h_ec_mc, g_t.unsqueeze(0), device)
                if a_ideal is not None and min_dist <= HC_RETRIEVAL_MAX_DIST:
                    a_star = action_buffer.retrieve(a_ideal, device)
                    step_losses.append(
                        HC_KL_WEIGHT * torch.sum((a_t.squeeze(0) - a_star.detach()) ** 2)
                    )

        with torch.no_grad():
            a_np  = a_t.squeeze(0).detach().cpu().numpy()
            gd    = (g_t - pos).cpu().numpy()
            cross = gd[0] * a_np[1] - gd[1] * a_np[0]
            dot   = a_np[0] * gd[0] + a_np[1] * gd[1]
            alpha = float(np.degrees(np.arctan2(cross, dot)))
            angle_seq.append(alpha)
            if t == 0:
                first_angle = alpha
                first_a     = a_t.squeeze(0).detach()

        # EC update — stop when lesioned (z_t stays 0)
        with torch.no_grad():
            if use_ec and not ec_lesion:
                s_tm1   = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                cb_pred = cb_frozen(pos.unsqueeze(0), a_t.detach(),
                                    z_t if z_t is not None else torch.zeros(1, LATENT_DIM, device=device))
                eps = s_next.detach().unsqueeze(0) - cb_pred
                z_t, h_ec = ec.step(
                    torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec
                )

        prev_a = a_t.detach()
        pos    = s_next.detach()
        vel    = vel_next
        steps += 1

    dist_norm  = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
    mean_angle = float(np.mean(angle_seq)) if angle_seq else 0.0
    first_angle = first_angle if first_angle is not None else 0.0

    loss = (torch.stack(step_losses).sum()
            if step_losses else torch.tensor(0.0, device=device, requires_grad=True))

    return loss, float(success), steps, dist_norm, mean_angle, first_angle, h_ec.detach(), first_a, g_t.detach()


# ── Training loop ─────────────────────────────────────────────────────────────
def run_babab(ec, cb_frozen, mc, device, seed=0,
              use_ec=True, use_cb=False, cb_noec=None,
              use_hc=False, ec_lesion=False):
    ec.eval()
    cb_frozen.eval()
    mc.train()
    for p in ec.parameters():        p.requires_grad_(False)
    for p in cb_frozen.parameters(): p.requires_grad_(False)

    params = list(mc.parameters())
    if cb_noec is not None:
        cb_noec.train()
        params += list(cb_noec.parameters())

    optimizer = torch.optim.Adam(params, lr=LR_MC)
    rng       = np.random.default_rng(seed)

    hc            = Hippocampus() if use_hc else None
    action_buffer = ActionBuffer() if use_hc else None

    curve = {
        "ep": [], "success": [], "steps": [], "dist_norm": [],
        "angle": [], "first_angle": [], "phase": [],
        "probe_b_ep": [], "probe_b_succ": [],
    }

    ep_global = 0

    for phase_idx, ctx in enumerate(PHASES):
        label = PHASE_LABELS[phase_idx]
        print(f"\n  Phase {label} ({N_PHASE_EP} episodes)...")

        # Accumulators for HC writing at end of phase
        hc_h_ec_acc = []
        hc_a_acc    = []
        hc_goal_acc = []

        for ep in range(1, N_PHASE_EP + 1):
            ep_global += 1

            mc.train()
            optimizer.zero_grad()

            loss, succ, steps, dist_norm, mean_angle, first_angle, h_ec_fin, first_a, ep_goal = run_episode(
                ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
                ctx=ctx, rng=rng, train=True,
                use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec,
                hc=hc, action_buffer=action_buffer,
                use_hc=use_hc, ec_lesion=ec_lesion,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            optimizer.step()

            # Push successful first-step actions to buffer
            if use_hc and succ and first_a is not None:
                action_buffer.push(first_a)
                # Accumulate for HC ideal (last HC_N_IDEAL successful eps)
                if ep > N_PHASE_EP - HC_N_IDEAL:
                    hc_h_ec_acc.append(h_ec_fin)
                    hc_a_acc.append(first_a)
                    hc_goal_acc.append(ep_goal)

            # Eval episode
            mc.eval()
            if cb_noec is not None:
                cb_noec.eval()
            with torch.no_grad():
                _, succ_e, steps_e, dist_e, ang_e, fang_e, _, _, _ = run_episode(
                    ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
                    ctx=ctx, rng=rng, train=False,
                    use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec,
                    hc=hc, action_buffer=action_buffer,
                    use_hc=use_hc, ec_lesion=ec_lesion,
                )

            curve["ep"].append(ep_global)
            curve["success"].append(float(succ_e))
            curve["steps"].append(float(steps_e))
            curve["dist_norm"].append(float(dist_e))
            curve["angle"].append(float(ang_e))
            curve["first_angle"].append(float(fang_e))
            curve["phase"].append(phase_idx)

            # Probe: during B phase, test on A context
            if phase_idx == B_IDX:
                with torch.no_grad():
                    _, probe_s, _, _, _, _, _, _, _ = run_episode(
                        ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
                        ctx=CTX_A, rng=rng, train=False,
                        use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec,
                        hc=hc, action_buffer=action_buffer,
                        use_hc=use_hc, ec_lesion=ec_lesion,
                    )
                curve["probe_b_ep"].append(ep_global)
                curve["probe_b_succ"].append(float(probe_s))

            if ep % 500 == 0:
                print(f"    ep {ep:4d} | acc(last100)={np.mean(curve['success'][-100:]):.2f}"
                      f" | angle(last100)={np.mean(curve['angle'][-100:]):.1f}°")

        # Write to HC at end of phase
        if use_hc and hc is not None and hc_h_ec_acc:
            hc.write(hc_h_ec_acc, hc_a_acc, label, hc_goal_acc)
            print(f"  HC wrote memory for '{label}' "
                  f"({len(hc.h_ec_mem)} memories stored)")

    return curve


# ── Plot helpers ──────────────────────────────────────────────────────────────
def smooth(x, w=SW):
    x = np.array(x)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fit_asymptote(x, y):
    from scipy.optimize import curve_fit
    def model(xv, L, y0, k):
        return L + (y0 - L) * np.exp(-k * xv)
    try:
        x, y  = np.asarray(x, float), np.asarray(y, float)
        p0    = [float(y[-1]), float(y[0]), 1e-3]
        popt, _ = curve_fit(model, x, y, p0=p0, maxfev=5000)
        L, y0, k = popt
        x95 = -np.log(0.05) / k if (k > 0 and abs(L - y0) > 1e-6) else None
        return L, y0, k, x95
    except Exception:
        return None, None, None, None


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_all(curves):
    os.makedirs(FIG_DIR, exist_ok=True)

    N_CONDS  = len(CONDITIONS)
    ep_local = np.arange(1, N_PHASE_EP + 1)
    SW       = 20

    def split_phases(arr):
        arr = np.array(arr)
        return [arr[i * N_PHASE_EP:(i + 1) * N_PHASE_EP] for i in range(len(PHASES))]

    # ── Fig 1: Learning curves (A / B / A2 only) ──────────────────────────────
    PLOT_PHASES = [A_IDX, B_IDX, A2_IDX]
    fig, axes  = plt.subplots(N_CONDS, len(PLOT_PHASES), figsize=(15, 4 * N_CONDS), sharey=True)

    for row, (cond, color, curve) in enumerate(zip(CONDITIONS, COND_COLORS, curves)):
        phases = split_phases(curve["success"])
        for col, ph_idx in enumerate(PLOT_PHASES):
            ax     = axes[row, col]
            y_raw  = np.array(phases[ph_idx])
            y_sm   = smooth(y_raw, SW)
            ep_sm  = ep_local[SW - 1:]

            ax.plot(ep_sm, y_sm, color=color, lw=2)
            L, y0, k, x95 = fit_asymptote(ep_sm, y_sm)
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
                ax.set_ylabel(f"{cond['name']}\nSuccess", fontsize=8)
            if row == N_CONDS - 1:
                ax.set_xlabel("Episode within phase", fontsize=8)

    fig.suptitle("HC-PPO (goal-conditioned) savings — success curves (A / B / A2)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hc_ppo_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved hc_ppo_learning_curves.png")

    # ── Fig 2: A vs A2 first-angle overlay ────────────────────────────────────
    EP_SHOW = 400
    fig, axes = plt.subplots(N_CONDS, 1, figsize=(6, 4 * N_CONDS), squeeze=False)

    for row, (cond, color, curve) in enumerate(zip(CONDITIONS, COND_COLORS, curves)):
        ax     = axes[row, 0]
        phases = split_phases(curve["first_angle"])
        for ph_idx, alpha, lbl in [
            (A_IDX,  0.35, PHASE_LABELS[A_IDX] + " (1st)"),
            (A2_IDX, 1.00, PHASE_LABELS[A2_IDX] + " (2nd)"),
        ]:
            y_raw = np.array(phases[ph_idx])
            y_sm  = smooth(y_raw, SW)
            ep_sm = ep_local[SW - 1:]
            mask  = ep_sm <= EP_SHOW
            ax.plot(ep_sm[mask], y_sm[mask], color=color, lw=2, alpha=alpha, label=lbl)

        ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° target")
        ax.set_xlim(1, EP_SHOW)
        ax.set_ylim(-110, 60)
        ax.legend(fontsize=7, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.set_title("A vs A2 — first-step angle", fontsize=10)
        ax.set_ylabel(f"{cond['name']}\nAngle (°)", fontsize=8)
        if row == N_CONDS - 1:
            ax.set_xlabel("Episode within phase", fontsize=8)

    fig.suptitle("HC-PPO (goal-conditioned): A vs A2 angle overlay\n(faded=1st, solid=2nd)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hc_ppo_angle_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved hc_ppo_angle_overlay.png")

    # ── Fig 3: Savings bar (normalised A episodes to 95%) ─────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    x95_A, x95_A2 = [], []
    for curve in curves:
        phases = split_phases(curve["success"])
        for ph_idx, store in [(A_IDX, x95_A), (A2_IDX, x95_A2)]:
            y_sm   = smooth(np.array(phases[ph_idx]), SW)
            ep_sm  = ep_local[SW - 1:]
            _, _, _, x95 = fit_asymptote(ep_sm, y_sm)
            store.append(x95 if (x95 is not None and x95 < N_PHASE_EP) else N_PHASE_EP)

    width = 0.35
    xs    = np.arange(N_CONDS)
    for i, (cond, color, a, a2) in enumerate(zip(CONDITIONS, COND_COLORS, x95_A, x95_A2)):
        norm_a2 = a2 / a if a > 0 else 1.0
        ax.bar(i - width / 2, 1.0,     width, color=color, alpha=0.4)
        ax.bar(i + width / 2, norm_a2, width, color=color, alpha=1.0)
        ax.annotate("", xy=(i + width / 2, norm_a2), xytext=(i + width / 2, 1.0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.text(i + width / 2 + 0.08, (1.0 + norm_a2) / 2,
                f"{norm_a2:.2f}", fontsize=8, va="center")

    ax.set_xticks(xs)
    ax.set_xticklabels([c["name"] for c in CONDITIONS], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Episodes to 95% asymptote\n(normalised to A = 1.0)", fontsize=10)
    ax.set_title("HC-PPO (goal-conditioned) Savings: A vs A2\n(faded=A=1.0, solid=A2 normalised)", fontsize=10)
    ax.set_ylim(0, 1.2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="A (1st)"),
                       Patch(facecolor="gray", alpha=1.0, label="A2 (2nd)")], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hc_ppo_savings_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved hc_ppo_savings_bar.png")

    # ── Fig 4: EC lesion comparison (MC+EC+HC vs lesion) ──────────────────────
    # Fig 4: lesion comparison — intact vs EC-only lesion vs EC+HC lesion
    name_map   = {c["name"]: i for i, c in enumerate(CONDITIONS)}
    intact_key = "MC + EC + CB + HC"
    lesion_pairs = [
        ("EC lesion (HC active)", "EC lesion only\n(HC gets wrong key)"),
        ("EC+HC lesion",          "EC+HC lesion\n(HC disabled)"),
    ]

    if intact_key in name_map:
        ii  = name_map[intact_key]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, ph_idx, ph_lbl in [(axes[0], A_IDX,  PHASE_LABELS[A_IDX]),
                                    (axes[1], A2_IDX, PHASE_LABELS[A2_IDX])]:
            # intact condition
            phases = split_phases(curves[ii]["first_angle"])
            y_sm   = smooth(np.array(phases[ph_idx]), SW)
            ep_sm  = ep_local[SW - 1:]
            mask   = ep_sm <= 400
            ax.plot(ep_sm[mask], y_sm[mask], color=COND_COLORS[ii],
                    lw=2, alpha=1.0, ls="-", label="EC intact (HC active)")

            # lesion conditions
            for (lkey, llabel), ls, alpha in zip(lesion_pairs, ["--", ":"], [0.8, 0.8]):
                if lkey in name_map:
                    li     = name_map[lkey]
                    phases = split_phases(curves[li]["first_angle"])
                    y_sm   = smooth(np.array(phases[ph_idx]), SW)
                    ax.plot(ep_sm[mask], y_sm[mask], color=COND_COLORS[li],
                            lw=2, alpha=alpha, ls=ls, label=llabel)

            ax.axhline(-60, color="gray", ls=":", lw=1.2, label="-60° target")
            ax.set_xlim(1, 400)
            ax.set_ylim(-110, 60)
            ax.set_title(ph_lbl, fontsize=10)
            ax.set_xlabel("Episode within phase", fontsize=9)
            ax.set_ylabel("First-step angle (°)", fontsize=9)
            ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle("EC lesion comparison — MC + EC + CB + HC (goal-conditioned)\n"
                     "solid=intact | dashed=EC lesion (HC active) | dotted=EC+HC lesion",
                     fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/hc_ppo_ec_lesion.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved hc_ppo_ec_lesion.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_seed(seed, ec, cb_frozen, device):
    seed_dir = os.path.join(FIG_DIR, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    data_path = f"{seed_dir}/hc_ppo_data.npz"

    if os.path.exists(data_path):
        print(f"Seed {seed}: data exists, loading for plotting.")
        d = np.load(data_path, allow_pickle=True)
        curves_all = list(d["curves"])
        plot_seed(seed, curves_all, seed_dir)
        return

    curves = []
    for cond in CONDITIONS:
        print(f"\n{'=' * 60}")
        print(f"Seed {seed} | Condition: {cond['name']}")
        print(f"{'=' * 60}")

        mc      = MotorCortex(use_ec=cond["use_ec"], use_cb=cond["use_cb"]).to(device)
        cb_noec = CerebellumCB_NoEC().to(device) if cond["retrain_cb_noec"] else None

        curve = run_babab(
            ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
            seed=seed,
            use_ec=cond["use_ec"],
            use_cb=cond["use_cb"],
            cb_noec=cb_noec,
            use_hc=cond["use_hc"],
            ec_lesion=cond["ec_lesion"],
        )
        curves.append(curve)

    np.savez(data_path, curves=curves)
    print(f"\nSeed {seed}: saved to {data_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(CKPT, map_location=device)

    ec        = EntorhinalCortex().to(device)
    cb_frozen = CerebellumCB().to(device)

    ec.load_state_dict(ckpt["ec"])

    cb_sd  = ckpt["cb"]
    remap  = {"net.0": "layer1.0", "net.2": "layer2.0", "net.4": "out"}
    new_sd = {}
    for k, v in cb_sd.items():
        prefix = ".".join(k.split(".")[:2])
        new_sd[k.replace(prefix, remap[prefix]) if prefix in remap else k] = v
    cb_frozen.load_state_dict(new_sd)

    ec.eval()
    cb_frozen.eval()
    print(f"Loaded EC + CB from {CKPT}")

    os.makedirs(FIG_DIR, exist_ok=True)

    # ── Multi-seed runs ────────────────────────────────────────────────────────
    seeds = [0, 1, 2, 3, 4, 5]   # seed 0 = original, 1-5 = new
    for seed in seeds:
        run_seed(seed, ec, cb_frozen, device)

    print("All seeds done.")


if __name__ == "__main__":
    main()

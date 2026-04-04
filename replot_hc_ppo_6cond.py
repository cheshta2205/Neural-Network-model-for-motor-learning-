"""
Replot HC-PPO figures with 6 conditions (drop 'EC lesion (HC active)', index 5).
Loads existing hc_ppo_data.npz — no retraining needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

DATA_PATH = "../../figs/ECtoCB/MC_EC_CB/MC_savings_norouting_60curl60_BABAB/hc_ppo/hc_ppo_data.npz"
FIG_DIR   = "../../figs/ECtoCB/MC_EC_CB/MC_savings_norouting/PPO/hc_ppo"

N_PHASE_EP   = 2000
N_PHASES     = 5
A_IDX, B_IDX, A2_IDX = 1, 2, 3
PHASE_LABELS = ["pre-B: curl+", "A: rot+60°", "B: curl+", "A2: rot+60°", "B2: curl+"]
SW           = 50

# ── Keep indices 0-4 and 6 (drop index 5 = EC lesion HC active) ──────────────
KEEP_IDX = [0, 1, 2, 3, 4, 6]
COND_NAMES_ALL = [
    "MC only",
    "MC + CB",
    "MC + EC",
    "MC + EC + CB",
    "MC + EC + CB + HC",
    "EC lesion (HC active)",   # dropped
    "EC+HC lesion",
]
COLORS_ALL = ["dimgray", "steelblue", "seagreen", "tomato", "darkorange", "salmon", "lightcoral"]

CONDITIONS  = [{"name": COND_NAMES_ALL[i]} for i in KEEP_IDX]
COND_COLORS = [COLORS_ALL[i] for i in KEEP_IDX]
N_CONDS     = len(CONDITIONS)

ep_local = np.arange(1, N_PHASE_EP + 1)


def smooth(x, w=SW):
    x = np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x


def fit_asymptote(x, y):
    def model(xv, L, y0, k):
        return L + (y0 - L) * np.exp(-k * xv)
    try:
        x, y = np.asarray(x, float), np.asarray(y, float)
        p0 = [float(y[-1]), float(y[0]), 1e-3]
        popt, _ = curve_fit(model, x, y, p0=p0, maxfev=5000)
        L, y0, k = popt
        x95 = -np.log(0.05) / k if (k > 0 and abs(L - y0) > 1e-6) else None
        return L, y0, k, x95
    except Exception:
        return None, None, None, None


def split_phases(arr):
    arr = np.array(arr)
    return [arr[i * N_PHASE_EP:(i + 1) * N_PHASE_EP] for i in range(N_PHASES)]


# ── Load & subset ─────────────────────────────────────────────────────────────
d         = np.load(DATA_PATH, allow_pickle=True)
all_curves = list(d["curves"])
curves    = [all_curves[i] for i in KEEP_IDX]

# ── Fig 1: Learning curves ────────────────────────────────────────────────────
fig, axes = plt.subplots(N_CONDS, 1, figsize=(12, 3 * N_CONDS), squeeze=False)
phase_boundaries = [i * N_PHASE_EP for i in range(1, N_PHASES)]

for row, (cond, color, curve) in enumerate(zip(CONDITIONS, COND_COLORS, curves)):
    ax   = axes[row, 0]
    succ = np.array(curve["success"])
    y_sm = smooth(succ)
    ep_g = np.arange(1, len(succ) + 1)
    ax.plot(ep_g[SW - 1:], y_sm, color=color, lw=1.5)
    for xb in phase_boundaries:
        ax.axvline(xb, color="gray", ls="--", lw=0.8)
    for i, lbl in enumerate(PHASE_LABELS):
        ax.text((i + 0.5) * N_PHASE_EP, 1.02, lbl, ha="center", fontsize=7,
                transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(cond["name"], fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if row == N_CONDS - 1:
        ax.set_xlabel("Episode (global)", fontsize=9)

fig.suptitle("HC-PPO Learning Curves — 6 conditions", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/hc_ppo_learning_curves_6cond.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved hc_ppo_learning_curves_6cond.png")

# ── Fig 2: A vs A2 angle overlay ─────────────────────────────────────────────
EP_SHOW = 400
fig, axes = plt.subplots(N_CONDS, 1, figsize=(6, 4 * N_CONDS), squeeze=False)

ANGLE_COLORS = {"A": "steelblue", "A2": "darkorange"}

for row, (cond, curve) in enumerate(zip(CONDITIONS, curves)):
    ax     = axes[row, 0]
    phases = split_phases(curve["angle"])
    for ph_idx, col, lbl in [
        (A_IDX,  ANGLE_COLORS["A"],  PHASE_LABELS[A_IDX] + " (1st)"),
        (A2_IDX, ANGLE_COLORS["A2"], PHASE_LABELS[A2_IDX] + " (2nd)"),
    ]:
        y_sm = smooth(np.array(phases[ph_idx]))
        ep_sm = ep_local[SW - 1:]
        mask  = ep_sm <= EP_SHOW
        ax.plot(ep_sm[mask], y_sm[mask], color=col, lw=2, alpha=1.0, label=lbl)
    ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° target")
    ax.set_xlim(1, EP_SHOW)
    ax.set_ylim(-110, 60)
    ax.legend(fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    if row == 0:
        ax.set_title("A vs A2 — mean trajectory angle", fontsize=10)
    ax.set_ylabel(f"{cond['name']}\nAngle (°)", fontsize=8)
    if row == N_CONDS - 1:
        ax.set_xlabel("Episode within phase", fontsize=8)

fig.suptitle("HC-PPO: A vs A2 angle overlay (6 conditions)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/hc_ppo_angle_overlay_6cond.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved hc_ppo_angle_overlay_6cond.png")

# ── Fig 3: Savings bar ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
x95_A, x95_A2 = [], []
for curve in curves:
    phases = split_phases(curve["success"])
    for ph_idx, store in [(A_IDX, x95_A), (A2_IDX, x95_A2)]:
        y_sm  = smooth(np.array(phases[ph_idx]))
        ep_sm = ep_local[SW - 1:]
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
ax.set_title("HC-PPO Savings: A vs A2 — 6 conditions\n(faded=A=1.0, solid=A2 normalised)", fontsize=10)
ax.set_ylim(0, 1.5)
ax.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="A (1st)"),
                   Patch(facecolor="gray", alpha=1.0, label="A2 (2nd)")], fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/hc_ppo_savings_bar_6cond.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved hc_ppo_savings_bar_6cond.png")

# ── Fig 4: Lesion comparison — intact HC vs EC+HC lesion ─────────────────────
intact_name = "MC + EC + CB + HC"
lesion_name = "EC+HC lesion"
name_to_curve = {cond["name"]: curve for cond, curve in zip(CONDITIONS, curves)}
name_to_color = {cond["name"]: color for cond, color in zip(CONDITIONS, COND_COLORS)}

if intact_name in name_to_curve and lesion_name in name_to_curve:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ph_idx, ph_lbl in [(axes[0], A_IDX, PHASE_LABELS[A_IDX]),
                                (axes[1], A2_IDX, PHASE_LABELS[A2_IDX])]:
        for name, ls, lw, alpha in [
            (intact_name, "-",  2.0, 1.0),
            (lesion_name, "--", 2.0, 0.8),
        ]:
            phases = split_phases(name_to_curve[name]["first_angle"])
            y_sm   = smooth(np.array(phases[ph_idx]))
            ep_sm  = ep_local[SW - 1:]
            mask   = ep_sm <= 400
            ax.plot(ep_sm[mask], y_sm[mask], color=name_to_color[name],
                    lw=lw, alpha=alpha, ls=ls, label=name)
        ax.axhline(-60, color="gray", ls=":", lw=1.2, label="-60° target")
        ax.set_xlim(1, 400)
        ax.set_ylim(-110, 60)
        ax.set_title(ph_lbl, fontsize=10)
        ax.set_xlabel("Episode within phase", fontsize=9)
        ax.set_ylabel("Mean trajectory angle (°)", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("EC+HC lesion vs intact\nsolid=MC+EC+CB+HC intact | dashed=EC+HC lesion", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hc_ppo_ec_lesion_6cond.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved hc_ppo_ec_lesion_6cond.png")

# ── Fig 4b: All 4 overlaid in one panel ──────────────────────────────────────
if intact_name in name_to_curve and lesion_name in name_to_curve:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, ph_idx, col, ls, lbl in [
        (intact_name, A_IDX,  "steelblue",  "-",  "Intact — A (1st)"),
        (intact_name, A2_IDX, "darkorange", "-",  "Intact — A2 (2nd)"),
        (lesion_name, A_IDX,  "steelblue",  "--", "Lesion — A (1st)"),
        (lesion_name, A2_IDX, "darkorange", "--", "Lesion — A2 (2nd)"),
    ]:
        phases = split_phases(name_to_curve[name]["first_angle"])
        y_sm   = smooth(np.array(phases[ph_idx]))
        ep_sm  = ep_local[SW - 1:]
        mask   = ep_sm <= EP_SHOW
        ax.plot(ep_sm[mask], y_sm[mask], color=col, ls=ls, lw=2, label=lbl)
    ax.axhline(-60, color="gray", ls=":", lw=1.2, label="-60° target")
    ax.set_xlim(1, EP_SHOW)
    ax.set_ylim(-110, 60)
    ax.set_xlabel("Episode within phase", fontsize=9)
    ax.set_ylabel("Mean trajectory angle (°)", fontsize=9)
    ax.set_title("Intact vs EC+HC lesion — A & A2 angle overlay", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hc_ppo_ec_lesion_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved hc_ppo_ec_lesion_overlay.png")


SW_ACC = 30  # smoothing for accuracy plots

def exp_model(xv, L, y0, k):
    return L + (y0 - L) * np.exp(-k * xv)

# ── Fig 5: ABA accuracy curves — intact vs EC+HC lesion ──────────────────────
if intact_name in name_to_curve and lesion_name in name_to_curve:
    fig, ax = plt.subplots(figsize=(9, 4))

    # Phases A(1), B(2), A2(3) → global episodes 2000-8000
    ABA_PHASES = [A_IDX, B_IDX, A2_IDX]
    intact_color = name_to_color[intact_name]
    lesion_color = name_to_color[lesion_name]

    for name, ls, lw, col in [
        (intact_name, "-",  2.0, "steelblue"),
        (lesion_name, "--", 2.0, "darkorange"),
    ]:
        succ = np.array(name_to_curve[name]["success"])
        # Take only phases 1-3 (episodes 2000-8000)
        aba_succ = succ[N_PHASE_EP:4 * N_PHASE_EP]
        y_sm = smooth(aba_succ, SW_ACC)
        ep_x = np.arange(1, len(y_sm) + 1)
        ax.plot(ep_x, y_sm, color=col, lw=lw, ls=ls, label=name)

    # Phase boundary between A/B and B/A2
    ax.axvline(N_PHASE_EP - SW + 1, color="gray", ls="--", lw=0.8)
    ax.axvline(2 * N_PHASE_EP - SW + 1, color="gray", ls="--", lw=0.8)
    for i, lbl in enumerate(["A: rot+60°", "B: curl+", "A2: rot+60°"]):
        ax.text((i + 0.5) * N_PHASE_EP, 1.02, lbl, ha="center", fontsize=8,
                transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Episode within ABA", fontsize=9)
    ax.set_ylabel("Success rate (smoothed)", fontsize=9)
    ax.set_title("ABA accuracy: intact vs EC+HC lesion", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hc_ppo_aba_accuracy_lesion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved hc_ppo_aba_accuracy_lesion.png")

# ── Fig 6: Savings comparison — A vs A2 accuracy (intact | lesioned) ─────────
SW_SAV = 30  # smoothing for savings accuracy plot (display + fit)
if intact_name in name_to_curve and lesion_name in name_to_curve:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, name, title in [
        (axes[0], intact_name, "Intact: MC+EC+CB+HC"),
        (axes[1], lesion_name, "EC+HC lesion"),
    ]:
        succ_phases = split_phases(name_to_curve[name]["success"])
        for ph_idx, col, lbl in [
            (A_IDX,  "steelblue",  "A (1st)"),
            (A2_IDX, "darkorange", "A2 (2nd)"),
        ]:
            y_raw = np.array(succ_phases[ph_idx])
            y_sm  = smooth(y_raw, SW_SAV)
            ep_sm = ep_local[SW_SAV - 1:]
            ax.plot(ep_sm, y_sm, color=col, lw=1.5, label=lbl)

            # asymptote fit using same SW_SAV
            L, y0, k, x95 = fit_asymptote(ep_sm, y_sm)
            if L is not None:
                y_fit = exp_model(ep_sm, L, y0, k)
                ax.plot(ep_sm, y_fit, color=col, lw=2, ls="--", alpha=0.9)
            if x95 is not None and 0 < x95 < N_PHASE_EP:
                ax.axvline(x95, color=col, lw=1.2, ls=":", alpha=0.8)
                ax.text(x95 + 15, 0.08, f"{x95:.0f}", color=col, fontsize=7, va="bottom")

        ax.set_ylim(0, 1.1)
        ax.set_xlim(1, N_PHASE_EP)
        ax.set_xlabel("Episode within phase", fontsize=9)
        ax.set_ylabel("Success rate", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Savings: A vs A2 accuracy — intact vs EC+HC lesion", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hc_ppo_savings_accuracy_lesion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved hc_ppo_savings_accuracy_lesion.png")


# ── Fig 6c: Savings accuracy — fit on raw, display smoothed at 60 ─────────────
SW_RAW = 60
if intact_name in name_to_curve and lesion_name in name_to_curve:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, name, title in [
        (axes[0], intact_name, "Intact: MC+EC+CB+HC"),
        (axes[1], lesion_name, "EC+HC lesion"),
    ]:
        succ_phases = split_phases(name_to_curve[name]["success"])
        for ph_idx, col, lbl in [
            (A_IDX,  "steelblue",  "A (1st)"),
            (A2_IDX, "darkorange", "A2 (2nd)"),
        ]:
            y_raw = np.array(succ_phases[ph_idx])
            # display: smoothed at 60
            y_sm  = smooth(y_raw, SW_RAW)
            ep_sm = ep_local[SW_RAW - 1:]
            ax.plot(ep_sm, y_sm, color=col, lw=1.5, label=lbl)
            # fit: raw data
            ep_raw = ep_local
            L, y0, k, x95 = fit_asymptote(ep_raw, y_raw)
            if L is not None:
                y_fit = exp_model(ep_raw, L, y0, k)
                ax.plot(ep_raw, y_fit, color=col, lw=2, ls="--", alpha=0.9)
            if x95 is not None and 0 < x95 < N_PHASE_EP:
                ax.axvline(x95, color=col, lw=1.2, ls=":", alpha=0.8)
                ax.text(x95 + 15, 0.08, f"{x95:.0f}", color=col, fontsize=7)

        ax.set_ylim(0, 1.1)
        ax.set_xlim(1, N_PHASE_EP)
        ax.set_xlabel("Episode within phase", fontsize=9)
        ax.set_ylabel("Success rate", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Savings: A vs A2 — fit on raw, display SW=60\nsolid=smoothed | dashed=exp fit", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hc_ppo_savings_raw_fit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved hc_ppo_savings_raw_fit.png")

# ── Fig 6d: Slope of angle descent over episodes — A vs A2, intact vs lesioned ─
if intact_name in name_to_curve and lesion_name in name_to_curve:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, name, title in [
        (axes[0], intact_name, "Intact: MC+EC+CB+HC"),
        (axes[1], lesion_name, "EC+HC lesion"),
    ]:
        angle_phases = split_phases(name_to_curve[name]["first_angle"])
        for ph_idx, col, lbl in [
            (A_IDX,  "steelblue",  "A (1st)"),
            (A2_IDX, "darkorange", "A2 (2nd)"),
        ]:
            y_sm  = smooth(np.array(angle_phases[ph_idx]), 20)
            ep_sm = ep_local[20 - 1:]
            # derivative: diff of smoothed angle, no further smoothing
            dy = np.diff(y_sm)
            dy_sm = smooth(dy, 20)
            ep_dy = ep_sm[:len(dy_sm)]

            mask  = ep_dy <= 150
            ax.plot(ep_dy[mask], dy_sm[mask], color=col, lw=1.5, label=lbl)

        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlim(1, 150)
        ax.set_xlabel("Episode within phase", fontsize=9)
        ax.set_ylabel("d(angle)/d(episode)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Slope of angle descent — A vs A2: intact vs EC+HC lesion\n(more negative = faster drop toward -60°)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hc_ppo_angle_slope.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved hc_ppo_angle_slope.png")

# ── Fig 6b: Slope of adaptation (k) — A vs A2, intact vs lesioned ─────────────
if intact_name in name_to_curve and lesion_name in name_to_curve:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, name, title in [
        (axes[0], intact_name, "Intact: MC+EC+CB+HC"),
        (axes[1], lesion_name, "EC+HC lesion"),
    ]:
        succ_phases = split_phases(name_to_curve[name]["success"])
        ks, x95s, lbls, cols = [], [], [], []
        for ph_idx, col, lbl in [
            (A_IDX,  "steelblue",  "A (1st)"),
            (A2_IDX, "darkorange", "A2 (2nd)"),
        ]:
            y_raw = np.array(succ_phases[ph_idx])
            y_sm  = smooth(y_raw, SW_SAV)
            ep_sm = ep_local[SW_SAV - 1:]
            L, y0, k, x95 = fit_asymptote(ep_sm, y_sm)
            ks.append(k if k is not None else 0)
            x95s.append(x95 if (x95 is not None and x95 < N_PHASE_EP) else N_PHASE_EP)
            lbls.append(lbl); cols.append(col)

        xs = np.arange(len(ks))
        bars = ax.bar(xs, ks, color=cols, width=0.5, alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(lbls, fontsize=9)
        for x, k_val in zip(xs, ks):
            ax.text(x, k_val + 0.0001, f"{k_val:.4f}", ha="center", fontsize=8)
        ax.set_ylabel("Adaptation rate (k)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Adaptation rate k — A vs A2: intact vs EC+HC lesion\n(higher k = faster learning)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hc_ppo_savings_slope.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved hc_ppo_savings_slope.png")

# ── Fig 7: Conditions × 3 — ABA accuracy + asymptote fit ─────────────────────
ABA_PHASE_INDICES = [A_IDX, B_IDX, A2_IDX]
ABA_PHASE_LABELS  = [PHASE_LABELS[i] for i in ABA_PHASE_INDICES]

fig, axes = plt.subplots(N_CONDS, 3, figsize=(12, 3 * N_CONDS), squeeze=False)


for row, (cond, color, curve) in enumerate(zip(CONDITIONS, COND_COLORS, curves)):
    succ_phases = split_phases(curve["success"])
    for col, ph_idx in enumerate(ABA_PHASE_INDICES):
        ax = axes[row, col]
        y_raw = np.array(succ_phases[ph_idx])
        y_sm  = smooth(y_raw, SW_ACC)
        ep_sm = ep_local[SW_ACC - 1:]

        ax.plot(ep_sm, y_sm, color=color, lw=1.8, label="accuracy")

        # asymptote fit — use SW=50 to avoid distorting the fit
        y_sm_fit = smooth(y_raw, SW)
        ep_sm_fit = ep_local[SW - 1:]
        L, y0, k, x95 = fit_asymptote(ep_sm_fit, y_sm_fit)
        if L is not None:
            y_fit = exp_model(ep_sm_fit, L, y0, k)
            ax.plot(ep_sm_fit, y_fit, color="black", lw=1.2, ls="--", alpha=0.7, label="fit")
            if x95 is not None and x95 < N_PHASE_EP:
                ax.axvline(x95, color="black", lw=1, ls=":", alpha=0.6)
                ax.text(x95 + 20, 0.1, f"x95={x95:.0f}", fontsize=6, color="black")

        ax.set_ylim(0, 1.1)
        ax.set_xlim(ep_sm[0], ep_sm[-1])
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.set_title(ABA_PHASE_LABELS[col], fontsize=9)
        if col == 0:
            ax.set_ylabel(f"{cond['name']}\nSuccess", fontsize=7)
        if row == N_CONDS - 1:
            ax.set_xlabel("Episode within phase", fontsize=8)

fig.suptitle("ABA accuracy + asymptote fit — all conditions", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/hc_ppo_aba_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved hc_ppo_aba_grid.png")

print("Done.")

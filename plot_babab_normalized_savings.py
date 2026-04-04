"""
Generate a normalised savings bar chart from saved BABAB data.
A = 1.0 (faded), A2 = x95_A2 / x95_A (solid). Lower bar = more savings.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Patch

BABAB_DIR   = "../../figs/ECtoCB/MC_EC_CB/MC_savings_norouting_60curl60_BABAB"
DATA_PATH   = f"{BABAB_DIR}/savings_4cond_data.npz"

N_PHASE_EP  = 2000
N_PHASES    = 5          # pre-B, A, B, A2, B2
A_IDX       = 1
A2_IDX      = 3

CONDITIONS  = [
    {"name": "MC only"},
    {"name": "MC + CB"},
    {"name": "MC + EC"},
    {"name": "MC + EC + CB"},
]
COND_COLORS = ["#555555", "#4C9BE8", "#4CAF50", "#E8734C"]


def smooth(x, w):
    x = np.array(x)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fit_asymptote(x, y):
    def model(xv, L, y0, k):
        return L + (y0 - L) * np.exp(-k * xv)
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
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


# ── Load data ─────────────────────────────────────────────────────────────────
d      = np.load(DATA_PATH, allow_pickle=True)
curves = list(d["curves"])

ep_local = np.arange(1, N_PHASE_EP + 1)
SW = 20

x95_A, x95_A2 = [], []
for curve in curves:
    succ_phases = split_phases(curve["success"])
    for phase_idx, store in [(A_IDX, x95_A), (A2_IDX, x95_A2)]:
        y_sm  = smooth(np.array(succ_phases[phase_idx]), SW)
        ep_sm = ep_local[SW - 1:SW - 1 + len(y_sm)]
        _, _, _, x95 = fit_asymptote(ep_sm, y_sm)
        val = x95 if (x95 is not None and x95 < N_PHASE_EP) else N_PHASE_EP
        store.append(val)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

width = 0.35
xs = np.arange(len(CONDITIONS))

for i, (cond, color, a, a2) in enumerate(zip(CONDITIONS, COND_COLORS, x95_A, x95_A2)):
    norm_a2 = a2 / a if a > 0 else 1.0
    ax.bar(i - width / 2, 1.0,     width, color=color, alpha=0.4)
    ax.bar(i + width / 2, norm_a2, width, color=color, alpha=1.0)
    saving = 1.0 - norm_a2
    ax.annotate(
        "", xy=(i + width / 2, norm_a2), xytext=(i + width / 2, 1.0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5)
    )
    ax.text(i + width / 2 + 0.08, (1.0 + norm_a2) / 2,
            f"{norm_a2:.2f}", fontsize=8, va="center")

ax.set_xticks(xs)
ax.set_xticklabels([c["name"] for c in CONDITIONS], fontsize=9, rotation=15, ha="right")
ax.set_ylabel("Episodes to 95% asymptote\n(normalised to A = 1.0)", fontsize=10)
ax.set_title("Savings: A vs A2 — BABAB, no routing\n(faded=A 1.0, solid=A2 normalised)", fontsize=10)
ax.set_ylim(0, 1.15)
ax.legend(
    handles=[Patch(facecolor="gray", alpha=0.4, label="A (1st)"),
             Patch(facecolor="gray", alpha=1.0, label="A2 (2nd)")],
    fontsize=9
)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = f"{BABAB_DIR}/savings_normalised_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

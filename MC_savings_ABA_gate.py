"""
MC_savings_ABA_gate.py

Same as MC_savings_ABA.py but with FiLM-based context-dependent gating in MC.

EC's latent z_ec produces scale and shift for every MC neuron (via a learned
linear layer), modulating which neurons are active per context. Neurons
amplified in context A receive gradient signal and update; suppressed neurons
are effectively frozen — protecting context B weights and vice versa.

FiLM reference: Perez et al. (2018), AAAI.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM  = 32
EC_HIDDEN   = 256
CB_HIDDEN   = 256
EC_INPUT    = 8
CB_PROJ_DIM = 32
MC_HIDDEN   = 256
MC_INPUT    = 2 + LATENT_DIM + CB_PROJ_DIM + 2   # s(2)+z_ec(32)+z_cb(32)+g(2) = 68

MAX_STEPS    = 20
N_PHASE_EP   = 2000
GOAL_RADIUS  = 0.1
GRID_SIZE    = 4.0
DT           = 1.0
LR_MC        = 1e-3
SMOOTH_WIN   = 1

CKPT    = "checkpoints/ectocb_8ctx_seq80_z32.pt"
FIG_DIR = "../../figs/ECtoCB/MC_policy/MC_savings_ABA_hardtask/newtask_withpre_gate"

# ── Phases ────────────────────────────────────────────────────────────────────
CTX_A  = (1.0,  np.pi/3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B  = (1.0,  0.0,     0.0, 0.0, 0.0, 0.0, "B: rot0")
PHASES = [CTX_B, CTX_A, CTX_B, CTX_A, CTX_B]
PHASE_LABELS = ["pre: rot 0°", "A: rot+60°", "B: rot 0°", "A2: rot+60°", "B2: rot 0°"]


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
        self.layer2 = nn.Sequential(nn.Linear(CB_HIDDEN, CB_HIDDEN),       nn.ReLU())
        self.out    = nn.Linear(CB_HIDDEN, 2)

    def forward(self, s, a, z):
        h = self.layer1(torch.cat([s, a, z], dim=-1))
        return self.out(self.layer2(h))

    def hidden(self, s, a, z):
        return self.layer2(self.layer1(torch.cat([s, a, z], dim=-1)))


class MotorCortex(nn.Module):
    """
    MC with FiLM gating.

    EC's z_ec (32-dim) is projected to scale + shift for each of the 256 MC
    neurons via self.film (Linear 32 → 512, split into two 256-dim vectors).

    Forward:
        h_new       = GRUCell(inp, h_mc)          # standard GRU hidden state
        scale, shift = film(z_ec).chunk(2)        # context-dependent gain/offset
        h_modulated = scale * h_new + shift        # FiLM modulation
        action      = tanh(readout(h_modulated))

    Gradient flow:
        d_loss/d_h_new[i] = scale[i] * d_loss/d_h_modulated[i]
    Neurons with scale ≈ 0 in the current context receive near-zero gradient
    → their weights are protected across context switches.
    """
    def __init__(self):
        super().__init__()
        self.hidden_dim = MC_HIDDEN
        self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.cell       = nn.GRUCell(MC_INPUT, MC_HIDDEN)
        self.film       = nn.Linear(LATENT_DIM, MC_HIDDEN * 2)  # → scale + shift
        self.readout    = nn.Linear(MC_HIDDEN, 2)
        W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
        self.register_buffer("cb_proj", W / W.norm(dim=0, keepdim=True))

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, h_ec, g, h_mc, cb_hidden):
        z_ec  = torch.tanh(self.ec_readout(h_ec))
        z_cb  = cb_hidden @ self.cb_proj
        inp   = torch.cat([s_t, z_ec, z_cb, g], dim=-1)

        h_new = self.cell(inp, h_mc)

        # FiLM: EC modulates gain and baseline of every MC neuron
        scale, shift = self.film(z_ec).chunk(2, dim=-1)
        h_modulated  = scale * h_new + shift

        return torch.tanh(self.readout(h_modulated)), h_new


# ── Differentiable env step ───────────────────────────────────────────────────
def env_step_torch(pos, vel, action, beta, theta, gamma, g, bx, by, device):
    cos_t = torch.tensor(float(np.cos(theta)), dtype=torch.float32, device=device)
    sin_t = torch.tensor(float(np.sin(theta)), dtype=torch.float32, device=device)
    R     = torch.stack([torch.stack([cos_t, -sin_t]),
                         torch.stack([sin_t,  cos_t])])
    curl  = torch.tensor([[0., -1.], [1., 0.]], device=device)
    b_vec = torch.tensor([float(bx), float(by)], device=device)
    F       = beta * (R @ action) + gamma * (curl @ vel) + g * b_vec
    vel_new = vel + F
    pos_new = torch.clamp(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new.detach()


# ── Single episode (train or eval) ───────────────────────────────────────────
def run_episode(ec, cb, mc, device, ctx, rng, train=True, ec_lesion=False):
    beta, theta, gamma, g, bx, by, _ = ctx

    start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(start - goal) < 0.5:
        goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    g_t    = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos    = torch.tensor(start, dtype=torch.float32, device=device)
    vel    = torch.zeros(2, device=device)
    h_ec   = ec.init_hidden(1, device)
    h_mc   = mc.init_hidden(1, device)
    z_t    = torch.zeros(1, LATENT_DIM, device=device)
    prev_a = torch.zeros(1, 2, device=device)

    step_losses  = []
    dist_seq     = []
    angle_seq    = []
    first_angle  = None
    success      = False
    steps        = 0
    init_dist    = torch.norm(pos - g_t).item()

    for t in range(MAX_STEPS):
        d = torch.norm(pos - g_t).item()
        dist_seq.append(d)
        if d < GOAL_RADIUS:
            success = True; break

        with torch.no_grad() if not train else torch.enable_grad():
            cb_h = cb.hidden(pos.unsqueeze(0), prev_a, z_t)

        h_ec_in = torch.zeros(1, EC_HIDDEN, device=device) if ec_lesion else h_ec
        a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec_in, g_t.unsqueeze(0), h_mc, cb_h)

        s_next, vel_next = env_step_torch(
            pos, vel, a_t.squeeze(0), beta, theta, gamma, g, bx, by, device)

        if train:
            step_losses.append(torch.sum((g_t - s_next) ** 2))

        with torch.no_grad():
            a_np  = a_t.squeeze(0).detach().cpu().numpy()
            gd    = (g_t - pos).cpu().numpy()
            cross = gd[0] * a_np[1] - gd[1] * a_np[0]
            dot   = a_np[0] * gd[0] + a_np[1] * gd[1]
            alpha = np.degrees(np.arctan2(cross, dot))
            angle_seq.append(alpha)
            if t == 0:
                first_angle = alpha

        with torch.no_grad():
            s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
            eps   = s_next.detach().unsqueeze(0) - cb(pos.unsqueeze(0), a_t.detach(), z_t)
            if not ec_lesion:
                z_t, h_ec = ec.step(
                    torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

        prev_a = a_t.detach()
        pos    = s_next.detach()
        vel    = vel_next
        steps += 1

    dist_norm   = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
    mean_angle  = np.mean(angle_seq) if angle_seq else 0.0
    first_angle = first_angle if first_angle is not None else 0.0

    loss = torch.stack(step_losses).sum() if step_losses else \
           torch.tensor(0.0, device=device, requires_grad=True)
    return loss, float(success), steps, dist_norm, mean_angle, first_angle


# ── ABA training loop ─────────────────────────────────────────────────────────
def run_aba(ec, cb, mc, device, seed=0, ec_lesion=False):
    ec.eval(); cb.eval(); mc.train()
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.Adam(mc.parameters(), lr=LR_MC)
    rng       = np.random.default_rng(seed)

    curve = {"ep": [], "success": [], "steps": [], "dist_norm": [], "angle": [], "first_angle": [], "phase": []}
    ep_global = 0

    for phase_idx, ctx in enumerate(PHASES):
        label = PHASE_LABELS[phase_idx]
        print(f"\nPhase {label} ({N_PHASE_EP} episodes)...")

        for ep in range(1, N_PHASE_EP + 1):
            ep_global += 1
            mc.train()

            optimizer.zero_grad()
            loss, _, _, _, _, _ = run_episode(ec, cb, mc, device, ctx, rng, train=True,
                                              ec_lesion=ec_lesion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            optimizer.step()

            mc.eval()
            with torch.no_grad():
                _, succ, steps, dist_norm, mean_angle, first_angle = run_episode(
                    ec, cb, mc, device, ctx, rng, train=False, ec_lesion=ec_lesion)
            curve["ep"].append(ep_global)
            curve["success"].append(float(succ))
            curve["steps"].append(steps)
            curve["dist_norm"].append(dist_norm)
            curve["angle"].append(mean_angle)
            curve["first_angle"].append(first_angle)
            curve["phase"].append(phase_idx)

            if ep % 500 == 0:
                recent_acc   = np.mean(curve["success"][-100:])
                recent_angle = np.mean(curve["angle"][-100:])
                print(f"  [{label}] ep {ep:4d} | acc (last 100)={recent_acc:.2f} | mean_angle={recent_angle:.1f}°")

    return curve


# ── Plots ─────────────────────────────────────────────────────────────────────
def smooth(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fit_asymptote(x, y):
    from scipy.optimize import curve_fit
    def model(x, L, y0, k):
        return L + (y0 - L) * np.exp(-k * x)
    try:
        y0_est = float(y[0])
        L_est  = float(y[-1])
        p0     = [L_est, y0_est, 1e-3]
        popt, _ = curve_fit(model, x, y, p0=p0, maxfev=5000)
        L, y0, k = popt
        if k > 0 and abs(L - y0) > 1e-6:
            x_95 = -np.log(0.05) / k
        else:
            x_95 = None
        return L, y0, k, x_95
    except Exception:
        return None, None, None, None


def plot_results(curve_intact, curve_lesion):
    os.makedirs(FIG_DIR, exist_ok=True)

    SW     = 20
    N_PHASES = len(PHASES)

    def split_phases(arr):
        arr = np.array(arr)
        return [arr[i*N_PHASE_EP:(i+1)*N_PHASE_EP] for i in range(N_PHASES)]

    succ_phases  = split_phases(curve_intact["success"])
    steps_phases = split_phases(curve_intact["steps"])
    dist_phases  = split_phases(curve_intact["dist_norm"])
    ep_local     = np.arange(1, N_PHASE_EP + 1)
    phase_boundaries = [i * N_PHASE_EP for i in range(1, N_PHASES)]

    # ── Fig: learning curves ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 4))
    for ax, phases, use_smooth, ylabel in zip(
        axes,
        [succ_phases, steps_phases, dist_phases],
        [True, True, False],
        ["Success rate", "Mean steps to goal", "Mean norm. dist to goal"],
    ):
        for phase_idx, y_raw in enumerate(phases):
            y = smooth(y_raw, SW) if use_smooth else y_raw
            ep_x = ep_local[SW-1:] if use_smooth else ep_local
            x_offset = phase_idx * N_PHASE_EP
            ax.plot(x_offset + ep_x, y, color="steelblue", lw=2)

            L, y0, k, x_95 = fit_asymptote(ep_x, y)
            print(f"  {ylabel} | {PHASE_LABELS[phase_idx]}: asymptote={L:.3f}, ep_95={int(x_95) if x_95 else 'N/A'}")
            if L is not None:
                ax.axhline(L, xmin=(x_offset) / (N_PHASES*N_PHASE_EP),
                           xmax=(x_offset + N_PHASE_EP) / (N_PHASES*N_PHASE_EP),
                           color="tomato", ls="--", lw=1.2)
            if x_95 is not None and x_95 < N_PHASE_EP:
                ax.axvline(x_offset + x_95, color="tomato", ls=":", lw=1.2)
                ax.text(x_offset + x_95, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 1.0,
                        f"ep {int(x_95)}", color="tomato", fontsize=7,
                        ha="center", va="bottom")

        for xb in phase_boundaries:
            ax.axvline(xb, color="gray", ls="--", lw=1)

        tick_positions, tick_labels = [], []
        for i in range(N_PHASES):
            for t in [0, 500, 1000, 1500, 2000]:
                tick_positions.append(i * N_PHASE_EP + t)
                tick_labels.append(str(t))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=6)

        y_vals = np.concatenate([np.array(p) for p in phases])
        y_top  = max(y_vals)
        for i, lbl in enumerate(PHASE_LABELS):
            mid = i * N_PHASE_EP + N_PHASE_EP // 2
            ax.text(mid, y_top, lbl, ha="center", va="bottom", fontsize=8,
                    color="dimgray",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))

        ax.set_xlabel("Episode within phase")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        if ylabel == "Success rate":
            ax.set_ylim(0, 1.05)

    fig.suptitle(f"ABAB savings | EC intact + FiLM gating\n"
                 f"A=rot+60°, B=rot 0°, A2=rot+60°, B2=rot 0°  "
                 f"(success/steps: rolling mean w={SW} | dist: raw | red=asymptote fit)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/aba_learning_curves2.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig: action compensation angle ────────────────────────────────────────
    angle_phases = split_phases(curve_intact["first_angle"])
    ep_local  = np.arange(1, N_PHASE_EP + 1)

    fig, ax = plt.subplots(figsize=(16, 4))
    for phase_idx, y_raw in enumerate(angle_phases):
        y        = np.array(y_raw)
        x_offset = phase_idx * N_PHASE_EP
        ax.plot(x_offset + ep_local, y, color="steelblue", lw=0.8, alpha=0.7)

    ax.axhline(-60, color="tomato",   ls="--", lw=1.2, label="−60° (target A/A2)")
    ax.axhline(0,   color="seagreen", ls="--", lw=1.2, label="0° (target B/B2 — no rotation)")

    for xb in phase_boundaries:
        ax.axvline(xb, color="gray", ls="--", lw=1)

    tick_positions, tick_labels = [], []
    for i in range(N_PHASES):
        for t in [0, 500, 1000, 1500, 2000]:
            tick_positions.append(i * N_PHASE_EP + t)
            tick_labels.append(str(t))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)

    for i, lbl in enumerate(PHASE_LABELS):
        mid = i * N_PHASE_EP + N_PHASE_EP // 2
        ax.text(mid, 170, lbl,
                ha="center", va="bottom", fontsize=8, color="dimgray",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))

    ax.set_xlabel("Episode within phase")
    ax.set_ylabel("First-step action angle relative to goal (degrees)")
    ax.set_title("First-step action angle — ABAB savings | EC intact + FiLM gating\nA=rot+60°, B=rot 0°  (no smoothing)")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/aba_action_angle.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig: savings overlay ──────────────────────────────────────────────────
    SW_OV = 10
    angle_phases = split_phases(curve_intact["first_angle"])
    ep_local = np.arange(1, N_PHASE_EP + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (idx1, idx2), target, title in zip(
        axes,
        [(1, 3), (2, 4)],
        [-60,      0],
        ["A vs A2  (rot+60°)", "B vs B2  (rot 0°)"],
    ):
        for idx, color, label in [(idx1, "steelblue", PHASE_LABELS[idx1]),
                                   (idx2, "tomato",    PHASE_LABELS[idx2])]:
            y_raw = np.array(angle_phases[idx])
            y_sm  = smooth(y_raw, SW_OV)
            ep_sm = ep_local[SW_OV - 1:]
            mask  = ep_sm <= 350
            ep_vis, y_vis = ep_sm[mask], y_sm[mask]
            ax.plot(ep_vis, y_vis, color=color, lw=2, alpha=0.4, label=label)
            L, y0, k, x_95 = fit_asymptote(ep_vis, y_vis)
            if L is not None and k is not None:
                y_fit = L + (y0 - L) * np.exp(-k * ep_vis)
                ax.plot(ep_vis, y_fit, color=color, ls="--", lw=2, zorder=3)

        ax.axhline(target, color="gray", ls="--", lw=1.2, label=f"{target:+d}° (target)")
        ax.set_xlabel("Episode within phase")
        ax.set_ylabel("First-step angle (deg)")
        ax.set_title(title)
        ax.set_xlim(1, 350)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Savings overlay — first-step action angle | FiLM gating\n"
                 f"Rolling mean w={SW_OV}  |  episodes 1–350  |  faster convergence = savings")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/aba_savings_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig: savings overlay w25, no asymptote ────────────────────────────────
    SW_OV25 = 25
    angle_phases = split_phases(curve_intact["first_angle"])
    ep_local = np.arange(1, N_PHASE_EP + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (idx1, idx2), target, title in zip(
        axes,
        [(1, 3), (2, 4)],
        [-60,      0],
        ["A vs A2  (rot+60°)", "B vs B2  (rot 0°)"],
    ):
        for idx, color, label in [(idx1, "steelblue", PHASE_LABELS[idx1]),
                                   (idx2, "tomato",    PHASE_LABELS[idx2])]:
            y_raw = np.array(angle_phases[idx])
            y_sm  = smooth(y_raw, SW_OV25)
            ep_sm = ep_local[SW_OV25 - 1:]
            mask  = ep_sm <= 350
            ep_vis, y_vis = ep_sm[mask], y_sm[mask]
            ax.plot(ep_vis, y_vis, color=color, lw=2, label=label)

        ax.axhline(target, color="gray", ls="--", lw=1.2, label=f"{target:+d}° (target)")
        ax.set_xlabel("Episode within phase")
        ax.set_ylabel("First-step angle (deg)")
        ax.set_title(title)
        ax.set_xlim(1, 350)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Savings overlay — first-step action angle | FiLM gating\n"
                 f"Rolling mean w={SW_OV25}  |  episodes 1–350  |  faster convergence = savings")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/aba_savings_overlay_w25.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig: ABA accuracy with asymptote ──────────────────────────────────────
    aba_indices  = [1, 2, 3]
    aba_labels   = [PHASE_LABELS[i] for i in aba_indices]
    succ_all     = split_phases(curve_intact["success"])
    SW_ACC       = 20
    ep_local_acc = np.arange(1, N_PHASE_EP + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, phase_idx, lbl in zip(axes, aba_indices, aba_labels):
        y_raw = np.array(succ_all[phase_idx])
        y_sm  = smooth(y_raw, SW_ACC)
        ep_sm = ep_local_acc[SW_ACC - 1:]
        ax.plot(ep_sm, y_sm, color="steelblue", lw=2, alpha=0.4, zorder=1)

        L, y0, k, x_95 = fit_asymptote(ep_sm, y_sm)
        if L is not None and k is not None:
            y_fit = L + (y0 - L) * np.exp(-k * ep_sm)
            ax.plot(ep_sm, y_fit, color="tomato", ls="--", lw=2,
                    label=f"asymptote={L:.2f}", zorder=3)
        if x_95 is not None and x_95 < N_PHASE_EP:
            ax.axvline(x_95, color="black", ls=":", lw=1.5, zorder=2)
            ax.text(x_95, L + 0.03 if L is not None else 0.5,
                    f"ep {int(x_95)}", color="black", fontsize=8,
                    ha="center", va="bottom")

        ax.set_xlim(1, N_PHASE_EP)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Episode within phase")
        ax.set_ylabel("Success rate")
        ax.set_title(lbl, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("ABA accuracy learning curves with asymptote fits | FiLM gating\n"
                 f"(extracted from BABAB run | rolling mean w={SW_ACC})")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/aba_acc_asymptote.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Figures saved to {FIG_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(CKPT, map_location=device)
    ec   = EntorhinalCortex().to(device)
    cb   = CerebellumCB().to(device)
    ec.load_state_dict(ckpt["ec"])

    cb_sd  = ckpt["cb"]
    remap  = {"net.0": "layer1.0", "net.2": "layer2.0", "net.4": "out"}
    new_sd = {}
    for k, v in cb_sd.items():
        prefix = ".".join(k.split(".")[:2])
        new_sd[k.replace(prefix, remap[prefix]) if prefix in remap else k] = v
    cb.load_state_dict(new_sd)
    ec.eval(); cb.eval()
    print(f"Loaded frozen EC + CB from {CKPT}")

    os.makedirs(FIG_DIR, exist_ok=True)
    data_path = f"{FIG_DIR}/aba_data.npz"

    if os.path.exists(data_path) and "--replot" in sys.argv:
        print(f"Loading saved data from {data_path} — skipping training.")
        d            = np.load(data_path, allow_pickle=True)
        curve_intact = d["curve_intact"].item()
        curve_lesion = d["curve_lesion"].item()
    else:
        mc_intact = MotorCortex().to(device)
        mc_lesion = MotorCortex().to(device)

        print("\n=== EC INTACT + FiLM ===")
        curve_intact = run_aba(ec, cb, mc_intact, device, seed=0, ec_lesion=False)
        print("\n=== EC LESION + FiLM ===")
        curve_lesion = run_aba(ec, cb, mc_lesion, device, seed=0, ec_lesion=True)

        torch.save(mc_intact.state_dict(), f"{FIG_DIR}/mc_intact.pt")
        torch.save(mc_lesion.state_dict(), f"{FIG_DIR}/mc_lesion.pt")
        np.savez(data_path, curve_intact=curve_intact, curve_lesion=curve_lesion)
        print(f"\nSaved checkpoints and data to {FIG_DIR}/")

    plot_results(curve_intact, curve_lesion)
    print("Done.")


if __name__ == "__main__":
    main()

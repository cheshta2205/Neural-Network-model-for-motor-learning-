"""
MC_savings_lesion.py

EC lesion experiment: adds 2 conditions to the existing 4-cond comparison.
Lesion = MoE architecture is kept (2 experts, pretrained thalamus) but
z_smooth is zeroed before the Thalamus → routing is blind to context.
EC still runs internally (feeds CB prediction errors) but the routing
signal is silenced.

Lesion conditions:
    5. MC+EC  (EC lesion) : MoE, no CB in MC, routing=blind
    6. MC+EC+CB (EC lesion): MoE, CB in MC,  routing=blind

Loads existing 4-cond data from data_4cond.npz.
Saves lesion data to data_lesion.npz.
Plots all 6 conditions together.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Hyperparameters (must match MC_savings_4cond.py) ──────────────────────────
LATENT_DIM   = 32
EC_HIDDEN    = 256
CB_HIDDEN    = 256
EC_INPUT     = 8
CB_PROJ_DIM  = 32
MC_HIDDEN    = 256
N_EXPERTS    = 2
TAU_START    = 1.0
TAU_END      = 0.1
EMA_ALPHA    = 0.5

MAX_STEPS    = 20
N_PHASE_EP   = 2000
N_PRETRAIN_EP = 500
GOAL_RADIUS  = 0.1
GRID_SIZE    = 4.0
DT           = 1.0
LR_MC        = 1e-3

CKPT          = "checkpoints/ectocb_8ctx_seq80_z32.pt"
CKPT_THALAMUS = "checkpoints/thalamus_pretrained_rot45curl_persistent_ema05.pt"
DATA_4COND    = "../../figs/ECtoCB/MC_policy/MC_savings_4cond/data_4cond.npz"
FIG_DIR       = "../../figs/ECtoCB/MC_policy/MC_savings_4cond"

CTX_A  = (1.0, np.pi/4, 0.0, 0.0, 0.0, 0.0, "A: rot+45")
CTX_B  = (1.0, 0.0,     1.0, 0.0, 0.0, 0.0, "B: curl+")
PHASES = [CTX_B, CTX_A, CTX_B, CTX_A, CTX_B]
PHASE_LABELS = ["pre: curl+", "A: rot+45°", "B: curl+", "A2: rot+45°", "B2: curl+"]

# Lesion conditions only
LESION_CONDITIONS = [
    {"name": "MC+EC (lesion)",     "use_cb_mc": False, "use_moe": True,  "ec_lesion": True},
    {"name": "MC+EC+CB (lesion)",  "use_cb_mc": True,  "use_moe": True,  "ec_lesion": True},
]

# All 6 for plotting
ALL_LABELS = [
    "MC only", "MC+CB", "MC+EC", "MC+EC+CB",
    "MC+EC (lesion)", "MC+EC+CB (lesion)",
]
ALL_COLORS = [
    "#888888", "#e07b54",        # MC only, MC+CB
    "#5b9bd5", "#2ca02c",        # MC+EC, MC+EC+CB
    "#5b9bd5", "#2ca02c",        # lesion versions — same color, dashed
]
ALL_LS = ["-", "-", "-", "-", "--", "--"]


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


class MCExpert(nn.Module):
    def __init__(self, mc_input):
        super().__init__()
        self.hidden_dim = MC_HIDDEN
        self.cell    = nn.GRUCell(mc_input, MC_HIDDEN)
        self.readout = nn.Linear(MC_HIDDEN, 2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, x, h):
        h_new = self.cell(x, h)
        return torch.tanh(self.readout(h_new)), h_new


class Thalamus(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(LATENT_DIM, N_EXPERTS)

    def forward(self, z_ec, temperature=1.0):
        logits = self.router(z_ec)
        return F.gumbel_softmax(logits, tau=temperature, hard=True)


class MotorCortex(nn.Module):
    def __init__(self, use_cb_mc=False, use_moe=False):
        super().__init__()
        self.use_cb_mc = use_cb_mc
        self.use_moe   = use_moe
        mc_input = 2 + (CB_PROJ_DIM if use_cb_mc else 0) + 2

        if use_moe:
            self.thalamus = Thalamus()
            self.experts  = nn.ModuleList([MCExpert(mc_input) for _ in range(N_EXPERTS)])
            W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
            self.register_buffer("cb_proj", W / W.norm(dim=0, keepdim=True))
        else:
            self.expert = MCExpert(mc_input)
            W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
            self.register_buffer("cb_proj", W / W.norm(dim=0, keepdim=True))

    def init_hidden(self, batch_size, device):
        if self.use_moe:
            return [e.init_hidden(batch_size, device) for e in self.experts]
        else:
            return self.expert.init_hidden(batch_size, device)

    def step(self, s_t, g, h, cb_hidden, z_smooth=None, temperature=1.0):
        z_cb = cb_hidden @ self.cb_proj

        if self.use_cb_mc:
            inp = torch.cat([s_t, z_cb, g], dim=-1)
        else:
            inp = torch.cat([s_t, g], dim=-1)

        if self.use_moe:
            z_ec_in = z_smooth if z_smooth is not None else \
                      torch.zeros(1, LATENT_DIM, device=s_t.device)
            weights = self.thalamus(z_ec_in, temperature=temperature)
            new_h, actions = [], []
            for i, expert in enumerate(self.experts):
                a_i, h_i = expert.step(inp, h[i])
                actions.append(a_i)
                new_h.append(h_i)
            actions_stack = torch.stack(actions, dim=1)
            action = (actions_stack * weights.unsqueeze(-1)).sum(dim=1)
            routing = weights[0, 1].item()
            return action, new_h, routing
        else:
            action, new_h = self.expert.step(inp, h)
            return action, new_h, 0.5


# ── Env step ──────────────────────────────────────────────────────────────────
def env_step_torch(pos, vel, action, beta, theta, gamma, g, bx, by, device):
    cos_t = torch.tensor(float(np.cos(theta)), dtype=torch.float32, device=device)
    sin_t = torch.tensor(float(np.sin(theta)), dtype=torch.float32, device=device)
    R     = torch.stack([torch.stack([cos_t, -sin_t]),
                         torch.stack([sin_t,  cos_t])])
    curl  = torch.tensor([[0., -1.], [1., 0.]], device=device)
    b_vec = torch.tensor([float(bx), float(by)], device=device)
    F_    = beta * (R @ action) + gamma * (curl @ vel) + g * b_vec
    vel_new = vel + F_
    pos_new = torch.clamp(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new.detach()


# ── Single episode ─────────────────────────────────────────────────────────────
def run_episode(ec, cb, mc, device, ctx, rng, train=True, temperature=1.0,
                h_ec_in=None, z_smooth_in=None, ec_lesion=False):
    """
    ec_lesion=True : EC runs internally (z_t updated) but z_smooth is zeroed
                     before being passed to Thalamus → routing is blind.
    """
    beta, theta, gamma, g, bx, by, _ = ctx

    start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(start - goal) < 0.5:
        goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    g_t      = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos      = torch.tensor(start, dtype=torch.float32, device=device)
    vel      = torch.zeros(2, device=device)
    h_ec     = h_ec_in     if h_ec_in     is not None else ec.init_hidden(1, device)
    h_mc     = mc.init_hidden(1, device)
    z_t      = torch.zeros(1, LATENT_DIM, device=device)
    z_smooth = z_smooth_in if z_smooth_in is not None else \
               torch.zeros(1, LATENT_DIM, device=device)
    prev_a   = torch.zeros(1, 2, device=device)

    step_losses = []
    dist_seq, angle_seq, routing_seq = [], [], []
    first_angle = None
    success = False
    init_dist = torch.norm(pos - g_t).item()

    for t in range(MAX_STEPS):
        d = torch.norm(pos - g_t).item()
        dist_seq.append(d)
        if d < GOAL_RADIUS:
            success = True; break

        with torch.no_grad():
            cb_h = cb.hidden(pos.unsqueeze(0), prev_a, z_t)

        with torch.no_grad():
            z_smooth = EMA_ALPHA * z_smooth + (1 - EMA_ALPHA) * z_t

        # Lesion: zero z_smooth before routing so Thalamus is blind
        z_in = None if ec_lesion else z_smooth

        a_t, h_mc, routing = mc.step(
            pos.unsqueeze(0), g_t.unsqueeze(0), h_mc, cb_h,
            z_smooth=z_in, temperature=temperature)
        routing_seq.append(routing)

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

        # EC always updates (lesion only silences routing, not internal EC)
        with torch.no_grad():
            s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
            eps   = s_next.detach().unsqueeze(0) - cb(pos.unsqueeze(0), a_t.detach(), z_t)
            z_t, h_ec = ec.step(
                torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

        prev_a = a_t.detach()
        pos    = s_next.detach()
        vel    = vel_next

    dist_norm    = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
    mean_angle   = np.mean(angle_seq)  if angle_seq  else 0.0
    first_angle  = first_angle if first_angle is not None else 0.0
    mean_routing = np.mean(routing_seq) if routing_seq else 0.5

    loss = torch.stack(step_losses).sum() if step_losses else \
           torch.tensor(0.0, device=device, requires_grad=True)
    return loss, float(success), dist_norm, mean_angle, first_angle, mean_routing, \
           h_ec.detach(), z_smooth.detach()


# ── Expert pre-training ────────────────────────────────────────────────────────
def pretrain_experts(ec, cb, mc, device, ec_lesion=False, seed=42):
    ec.eval(); cb.eval(); mc.train()
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [p for p in mc.parameters() if p.requires_grad], lr=LR_MC)
    rng = np.random.default_rng(seed)
    for ctx in [CTX_A, CTX_B]:
        for _ in range(N_PRETRAIN_EP):
            optimizer.zero_grad()
            loss, _, _, _, _, _, _, _ = run_episode(
                ec, cb, mc, device, ctx, rng,
                train=True, temperature=100.0, ec_lesion=ec_lesion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            optimizer.step()
    print(f"  Expert pre-training done.")


# ── ABA training loop ──────────────────────────────────────────────────────────
def run_aba(ec, cb, mc, device, ec_lesion=False, seed=0):
    ec.eval(); cb.eval(); mc.train()
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [p for p in mc.parameters() if p.requires_grad], lr=LR_MC)
    rng = np.random.default_rng(seed)
    total_eps = N_PHASE_EP * len(PHASES)

    curve = {"ep": [], "success": [], "dist_norm": [],
             "angle": [], "first_angle": [], "phase": [], "routing": []}
    ep_global = 0

    for phase_idx, ctx in enumerate(PHASES):
        label = PHASE_LABELS[phase_idx]
        print(f"  Phase {label}...")
        h_ec_carry, z_smooth_carry = None, None

        for ep in range(1, N_PHASE_EP + 1):
            ep_global += 1
            tau = TAU_START - (TAU_START - TAU_END) * (ep_global / total_eps)
            mc.train()
            optimizer.zero_grad()
            loss, _, _, _, _, _, h_ec_carry, z_smooth_carry = run_episode(
                ec, cb, mc, device, ctx, rng,
                train=True, temperature=tau,
                h_ec_in=h_ec_carry, z_smooth_in=z_smooth_carry,
                ec_lesion=ec_lesion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            optimizer.step()

            mc.eval()
            with torch.no_grad():
                _, succ, dist_norm, mean_angle, first_angle, routing, _, _ = run_episode(
                    ec, cb, mc, device, ctx, rng,
                    train=False, temperature=tau,
                    h_ec_in=h_ec_carry, z_smooth_in=z_smooth_carry,
                    ec_lesion=ec_lesion)

            curve["ep"].append(ep_global)
            curve["success"].append(float(succ))
            curve["dist_norm"].append(dist_norm)
            curve["angle"].append(mean_angle)
            curve["first_angle"].append(first_angle)
            curve["phase"].append(phase_idx)
            curve["routing"].append(routing)

            if ep % 500 == 0:
                print(f"    ep {ep:4d} | acc={np.mean(curve['success'][-100:]):.2f} | "
                      f"angle={np.mean(curve['angle'][-100:]):.1f}° | "
                      f"routing={np.mean(curve['routing'][-100:]):.2f} | τ={tau:.3f}")

    return curve


# ── Helpers ───────────────────────────────────────────────────────────────────
def smooth(x, w):
    if len(x) < w: return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fit_x95(y, ep_x):
    def model(x, L, y0, k): return L + (y0 - L) * np.exp(-k * x)
    try:
        p0 = [float(y[-1]), float(y[0]), 1e-3]
        popt, _ = curve_fit(model, ep_x, y, p0=p0, maxfev=5000)
        L, y0, k = popt
        x95 = -np.log(0.05) / k if k > 0 and abs(L - y0) > 1e-6 else None
        return L, x95
    except:
        return None, None


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_all(all_curves, all_labels, all_colors, all_ls):
    os.makedirs(FIG_DIR, exist_ok=True)
    N      = len(all_curves)
    N_PHASES_N = len(PHASES)
    SW     = 20
    ep_local = np.arange(1, N_PHASE_EP + 1)

    def split_phases(arr):
        arr = np.array(arr)
        return [arr[i * N_PHASE_EP:(i+1) * N_PHASE_EP] for i in range(N_PHASES_N)]

    # ── Learning curves (N×3) ─────────────────────────────────────────────────
    fig, axes = plt.subplots(N, 3, figsize=(20, 4 * N))
    metrics  = ["success", "dist_norm", "angle"]
    ylabels  = ["Success rate", "Norm. dist to goal", "Mean angle (deg)"]

    for ci, (curve, label, color, ls) in enumerate(
            zip(all_curves, all_labels, all_colors, all_ls)):
        for mi, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
            ax = axes[ci, mi]
            phases = split_phases(curve[metric])
            for pi, y_raw in enumerate(phases):
                y    = smooth(y_raw, SW)
                ep_x = ep_local[SW-1:]
                xoff = pi * N_PHASE_EP
                ax.plot(xoff + ep_x, y, color=color, lw=1.5, ls=ls)
                L, x95 = fit_x95(y, ep_x)
                y_min, y_max = np.min(y), np.max(y)
                y_rng = max(y_max - y_min, 1e-6)
                if L is not None and y_min - 2*y_rng < L < y_max + 2*y_rng:
                    ax.axhline(L, xmin=xoff/(N_PHASES_N*N_PHASE_EP),
                               xmax=(xoff+N_PHASE_EP)/(N_PHASES_N*N_PHASE_EP),
                               color="tomato", ls="--", lw=1)
                if x95 is not None and x95 < N_PHASE_EP:
                    ax.axvline(xoff + x95, color="tomato", ls=":", lw=1)

            for pi in range(1, N_PHASES_N):
                ax.axvline(pi * N_PHASE_EP, color="gray", ls="--", lw=0.8)
            if ci == 0:
                ax.set_title(ylabel, fontsize=10)
            if mi == 0:
                ax.set_ylabel(label, fontsize=9, color=color)
            if metric == "success":
                ax.set_ylim(0, 1.05)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("Episode", fontsize=7)
            if ci == 0:
                y_top = ax.get_ylim()[1]
                for pi, lbl in enumerate(PHASE_LABELS):
                    mid = pi * N_PHASE_EP + N_PHASE_EP // 2
                    ax.text(mid, y_top * 0.98, lbl, ha="center", va="top",
                            fontsize=7, color="dimgray")

    fig.suptitle("ABA savings — 4 conditions + EC lesion | rot+45° vs curl+",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/learning_curves_6cond.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved learning_curves_6cond.png")

    # ── Savings delta bar (x95_A - x95_A2) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x95_A, x95_A2 = [], []
    for curve in all_curves:
        succ = np.array(curve["success"])
        for phase_idx, store in [(1, x95_A), (3, x95_A2)]:
            y_raw = succ[phase_idx * N_PHASE_EP:(phase_idx+1) * N_PHASE_EP]
            y = smooth(y_raw, SW)
            ep_x = ep_local[SW-1:]
            _, x95 = fit_x95(y, ep_x)
            val = min(x95, N_PHASE_EP) if x95 is not None and x95 > 0 else N_PHASE_EP
            store.append(val)

    deltas = [a - a2 for a, a2 in zip(x95_A, x95_A2)]
    x = np.arange(N)
    bars = ax.bar(x, deltas, color=all_colors, alpha=0.85,
                  edgecolor='k', linewidth=0.8)
    for i in range(4, N):
        bars[i].set_hatch("//")
    ax.axhline(0, color='k', lw=1.0, ls='--')
    for i, dv in enumerate(deltas):
        yoff = 15 if dv >= 0 else -25
        va = 'bottom' if dv >= 0 else 'top'
        ax.text(i, dv + yoff, f"{int(dv)}", ha='center', va=va,
                fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel("x95(A) − x95(A2)  [episodes]", fontsize=10)
    ax.set_title("Savings: episodes saved in A2 vs A\n(positive = savings | hatched = EC lesion)")
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_delta_bar_6cond.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_delta_bar_6cond.png")

    # ── All-phase x95 table print ─────────────────────────────────────────────
    print(f"\n{'Condition':<22} {'pre-B':>8} {'A':>8} {'B':>8} {'A2':>8} {'B2':>8}")
    print("-" * 62)
    for ci, (curve, label) in enumerate(zip(all_curves, all_labels)):
        succ = np.array(curve["success"])
        row = []
        for ph in range(5):
            y_raw = succ[ph * N_PHASE_EP:(ph+1) * N_PHASE_EP]
            y = smooth(y_raw, SW)
            ep_x = ep_local[SW-1:]
            _, x95 = fit_x95(y, ep_x)
            val = int(min(x95, N_PHASE_EP)) if x95 is not None and x95 > 0 else N_PHASE_EP
            row.append(val)
        print(f"{label:<22} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8}")

    # ── acc asymptote — 6 rows (4 base + 2 lesion), A / B / A2 phases ─────────
    aba_phases = [1, 2, 3]  # A, B, A2
    aba_labels = [PHASE_LABELS[i] for i in aba_phases]
    SW_ACC = 20

    fig, axes = plt.subplots(N, 3, figsize=(15, 4 * N))
    for ci, (curve, label, color, ls) in enumerate(
            zip(all_curves, all_labels, all_colors, all_ls)):
        succ = np.array(curve["success"])
        for mi, (ph, ph_lbl) in enumerate(zip(aba_phases, aba_labels)):
            ax = axes[ci, mi]
            y_raw = succ[ph * N_PHASE_EP:(ph+1) * N_PHASE_EP]
            y_sm  = smooth(y_raw, SW_ACC)
            ep_sm = ep_local[SW_ACC - 1:]
            ax.plot(ep_sm, y_sm, color=color, lw=2, ls=ls, alpha=0.6)
            L, x95 = fit_x95(y_sm, ep_sm)
            if L is not None:
                y_fit_vals = [L + (y_sm[0] - L) * np.exp(-(-np.log(0.05)/x95 if x95 else 0) * e / (N_PHASE_EP/3))
                              for e in ep_sm]
                # just draw asymptote line
                ax.axhline(L, color=color, ls=":", lw=1, alpha=0.6)
                ax.text(N_PHASE_EP * 0.98, L + 0.02, f"L={L:.2f}",
                        ha="right", va="bottom", fontsize=7, color=color)
            if x95 is not None and 0 < x95 < N_PHASE_EP:
                ax.axvline(x95, color="black", ls=":", lw=1.2)
                ax.text(x95, 0.05, f"ep {int(x95)}", color="black",
                        fontsize=7, ha="center", va="bottom")
            ax.set_xlim(1, N_PHASE_EP)
            ax.set_ylim(0, 1.05)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("Episode", fontsize=7)
            if mi == 0:
                ax.set_ylabel(label, fontsize=8, color=color)
            if ci == 0:
                ax.set_title(ph_lbl, fontsize=10)
        # separator between base and lesion
        if ci == 3:
            for mi in range(3):
                axes[ci, mi].spines["bottom"].set_linewidth(2)

    fig.suptitle("ABA accuracy — 6 conditions (solid=EC intact, dashed=EC lesion)\n"
                 "rot+45° vs curl+ | dotted=x95 | horizontal dotted=asymptote",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_all_acc_asymptote_6cond.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_all_acc_asymptote_6cond.png")

    # ── Routing comparison (EC conditions only) ───────────────────────────────
    ec_indices = [2, 3, 4, 5]  # MC+EC, MC+EC+CB, and their lesions
    ec_labels  = [all_labels[i] for i in ec_indices]
    ec_colors  = [all_colors[i] for i in ec_indices]
    ec_ls      = [all_ls[i] for i in ec_indices]

    fig, axes = plt.subplots(1, len(PHASES), figsize=(18, 3), sharey=True)
    for pi, (ctx, ph_label) in enumerate(zip(PHASES, PHASE_LABELS)):
        ax = axes[pi]
        for idx, (ci, lbl, col, ls) in enumerate(
                zip(ec_indices, ec_labels, ec_colors, ec_ls)):
            curve = all_curves[ci]
            routing_raw = np.array(curve["routing"])
            ph_data = routing_raw[pi * N_PHASE_EP:(pi+1) * N_PHASE_EP]
            r_sm = smooth(ph_data, 50)
            ax.plot(r_sm, color=col, lw=1.5, ls=ls, label=lbl)
        ax.axhline(0.5, color='gray', ls=':', lw=1)
        ax.set_title(ph_label, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Episode", fontsize=7)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel("Routing (expert 1 weight)", fontsize=8)
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Thalamus routing — EC vs EC lesion conditions", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/routing_ec_vs_lesion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved routing_ec_vs_lesion.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load frozen EC + CB
    ckpt = torch.load(CKPT, map_location=device)
    ec   = EntorhinalCortex().to(device)
    ec.load_state_dict(ckpt["ec"])

    cb     = CerebellumCB().to(device)
    cb_sd  = ckpt["cb"]
    remap  = {"net.0": "layer1.0", "net.2": "layer2.0", "net.4": "out"}
    new_sd = {}
    for k, v in cb_sd.items():
        prefix = ".".join(k.split(".")[:2])
        new_sd[k.replace(prefix, remap[prefix]) if prefix in remap else k] = v
    cb.load_state_dict(new_sd)
    ec.eval(); cb.eval()
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)
    print(f"Loaded frozen EC + CB from {CKPT}")

    thal_ckpt = torch.load(CKPT_THALAMUS, map_location=device)
    os.makedirs(FIG_DIR, exist_ok=True)
    data_path = f"{FIG_DIR}/data_lesion.npz"

    if os.path.exists(data_path) and "--replot" in sys.argv:
        print("Loading saved lesion data — skipping training.")
        d      = np.load(data_path, allow_pickle=True)
        lesion_curves = list(d["curves"])
    else:
        lesion_curves = []
        for cond in LESION_CONDITIONS:
            print(f"\n=== {cond['name']} ===")
            mc = MotorCortex(use_cb_mc=cond["use_cb_mc"],
                             use_moe=cond["use_moe"]).to(device)
            # Load pretrained thalamus (frozen)
            mc.thalamus.router.load_state_dict({
                "weight": thal_ckpt["thalamus_state_dict"]["router.weight"],
                "bias":   thal_ckpt["thalamus_state_dict"]["router.bias"]})
            for p in mc.thalamus.router.parameters():
                p.requires_grad_(False)
            print(f"  Loaded + frozen pretrained thalamus.")

            # Pre-train with ec_lesion=True (routing blind during pretraining too)
            print(f"  Pre-training experts ({N_PRETRAIN_EP} eps/ctx)...")
            pretrain_experts(ec, cb, mc, device, ec_lesion=True, seed=42)

            curve = run_aba(ec, cb, mc, device, ec_lesion=True, seed=0)
            lesion_curves.append(curve)

        np.savez(data_path, curves=lesion_curves)
        print(f"\nSaved lesion data to {data_path}")

    # Load original 4-cond data
    print(f"\nLoading 4-condition data from {DATA_4COND}...")
    d4 = np.load(DATA_4COND, allow_pickle=True)
    base_curves = list(d4["curves"])

    all_curves = base_curves + lesion_curves
    plot_all(all_curves, ALL_LABELS, ALL_COLORS, ALL_LS)
    print("Done.")


if __name__ == "__main__":
    main()

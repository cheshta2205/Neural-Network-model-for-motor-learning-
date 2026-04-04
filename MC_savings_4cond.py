"""
MC_savings_4cond.py

4-condition ABA savings experiment with Thalamus-gated MoE for EC conditions.

Conditions:
    1. MC only       : 1 GRU, input [s_t, g],        no EC, no CB
    2. MC + CB       : 1 GRU, input [s_t, z_cb, g],  no EC, CB→MC
    3. MC + EC       : 2 GRU (MoE), input [s_t, g],  EC→Thalamus→routing, CB→EC errors only
    4. MC + EC + CB  : 2 GRU (MoE), input [s_t,z_cb,g], EC→Thalamus→routing, CB→MC+EC

EC always communicates to MC via Thalamus routing only — never directly in input.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Hyperparameters ────────────────────────────────────────────────────────────
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
FIG_DIR       = "../../figs/ECtoCB/MC_policy/MC_savings_4cond"

CTX_A  = (1.0, np.pi/4, 0.0, 0.0, 0.0, 0.0, "A: rot+45")
CTX_B  = (1.0, 0.0,     1.0, 0.0, 0.0, 0.0, "B: curl+")
PHASES = [CTX_B, CTX_A, CTX_B, CTX_A, CTX_B]
PHASE_LABELS = ["pre: curl+", "A: rot+45°", "B: curl+", "A2: rot+45°", "B2: curl+"]

CONDITIONS = [
    {"name": "MC only",      "use_ec": False, "use_cb_mc": False, "use_moe": False},
    {"name": "MC + CB",      "use_ec": False, "use_cb_mc": True,  "use_moe": False},
    {"name": "MC + EC",      "use_ec": True,  "use_cb_mc": False, "use_moe": True},
    {"name": "MC + EC + CB", "use_ec": True,  "use_cb_mc": True,  "use_moe": True},
]

COLORS = ["gray", "steelblue", "seagreen", "tomato"]


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
    """
    Unified MC: single expert (MC only / MC+CB) or MoE (MC+EC / MC+EC+CB).
    use_cb_mc : whether z_cb enters expert input
    use_moe   : whether Thalamus MoE is used
    """
    def __init__(self, use_cb_mc=False, use_moe=False):
        super().__init__()
        self.use_cb_mc = use_cb_mc
        self.use_moe   = use_moe
        mc_input = 2 + (CB_PROJ_DIM if use_cb_mc else 0) + 2  # s_t + [z_cb] + g

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
        z_cb = cb_hidden @ self.cb_proj  # (1, CB_PROJ_DIM)

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
                h_ec_in=None, z_smooth_in=None, use_ec=True):
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

        z_in = z_smooth if (use_ec and mc.use_moe) else None

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

        with torch.no_grad():
            s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
            eps   = s_next.detach().unsqueeze(0) - cb(pos.unsqueeze(0), a_t.detach(), z_t)
            if use_ec:
                z_t, h_ec = ec.step(
                    torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

        prev_a = a_t.detach()
        pos    = s_next.detach()
        vel    = vel_next

    dist_norm   = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
    mean_angle  = np.mean(angle_seq)  if angle_seq  else 0.0
    first_angle = first_angle if first_angle is not None else 0.0
    mean_routing = np.mean(routing_seq) if routing_seq else 0.5

    loss = torch.stack(step_losses).sum() if step_losses else \
           torch.tensor(0.0, device=device, requires_grad=True)
    return loss, float(success), dist_norm, mean_angle, first_angle, mean_routing, \
           h_ec.detach(), z_smooth.detach()


# ── Expert pre-training ────────────────────────────────────────────────────────
def pretrain_experts(ec, cb, mc, device, use_ec, seed=42):
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
                train=True, temperature=100.0, use_ec=use_ec)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            optimizer.step()
    print(f"  Expert pre-training done.")


# ── ABA training loop ──────────────────────────────────────────────────────────
def run_aba(ec, cb, mc, device, use_ec, seed=0):
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
                h_ec_in=h_ec_carry, z_smooth_in=z_smooth_carry, use_ec=use_ec)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            optimizer.step()

            mc.eval()
            with torch.no_grad():
                _, succ, dist_norm, mean_angle, first_angle, routing, _, _ = run_episode(
                    ec, cb, mc, device, ctx, rng,
                    train=False, temperature=tau,
                    h_ec_in=h_ec_carry, z_smooth_in=z_smooth_carry, use_ec=use_ec)

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
def plot_all(curves):
    os.makedirs(FIG_DIR, exist_ok=True)
    N_CONDS  = len(CONDITIONS)
    N_PHASES = len(PHASES)
    SW       = 20
    ep_local = np.arange(1, N_PHASE_EP + 1)

    def split_phases(arr):
        arr = np.array(arr)
        return [arr[i * N_PHASE_EP:(i+1) * N_PHASE_EP] for i in range(N_PHASES)]

    # ── 4x3 learning curves ───────────────────────────────────────────────────
    fig, axes = plt.subplots(N_CONDS, 3, figsize=(20, 4 * N_CONDS))
    metrics   = ["success", "dist_norm", "angle"]
    ylabels   = ["Success rate", "Norm. dist to goal", "Mean angle (deg)"]

    for ci, (cond, curve) in enumerate(zip(CONDITIONS, curves)):
        color = COLORS[ci]
        for mi, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
            ax = axes[ci, mi]
            phases = split_phases(curve[metric])
            for pi, y_raw in enumerate(phases):
                y    = smooth(y_raw, SW)
                ep_x = ep_local[SW-1:]
                xoff = pi * N_PHASE_EP
                ax.plot(xoff + ep_x, y, color=color, lw=1.5)
                # asymptote — clip to data range
                L, x95 = fit_x95(y, ep_x)
                y_min, y_max = np.min(y), np.max(y)
                y_rng = max(y_max - y_min, 1e-6)
                if L is not None and y_min - 2*y_rng < L < y_max + 2*y_rng:
                    ax.axhline(L, xmin=xoff/(N_PHASES*N_PHASE_EP),
                               xmax=(xoff+N_PHASE_EP)/(N_PHASES*N_PHASE_EP),
                               color="tomato", ls="--", lw=1)
                if x95 is not None and x95 < N_PHASE_EP:
                    ax.axvline(xoff + x95, color="tomato", ls=":", lw=1)

            for pi in range(1, N_PHASES):
                ax.axvline(pi * N_PHASE_EP, color="gray", ls="--", lw=0.8)
            if ci == 0:
                ax.set_title(ylabel, fontsize=10)
            if mi == 0:
                ax.set_ylabel(cond["name"], fontsize=9, color=color)
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

    fig.suptitle("ABA savings — 4 conditions | rot+45° vs curl+ | EC via Thalamus",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/learning_curves_4cond.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved learning_curves_4cond.png")

    # ── Savings bar chart (x95 A vs A2) ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    cond_names = [c["name"] for c in CONDITIONS]
    x95_A, x95_A2 = [], []

    for curve in curves:
        succ = np.array(curve["success"])
        for phase_idx, label in [(1, "A"), (3, "A2")]:
            y_raw = succ[phase_idx * N_PHASE_EP:(phase_idx+1) * N_PHASE_EP]
            y     = smooth(y_raw, SW)
            ep_x  = ep_local[SW-1:]
            _, x95 = fit_x95(y, ep_x)
            (x95_A if label == "A" else x95_A2).append(
                min(x95, N_PHASE_EP) if x95 else N_PHASE_EP)

    x = np.arange(N_CONDS)
    w = 0.35
    bars_A  = ax.bar(x - w/2, x95_A,  w, label="A (1st)",  color=COLORS, alpha=0.4)
    bars_A2 = ax.bar(x + w/2, x95_A2, w, label="A2 (2nd)", color=COLORS, alpha=0.9)
    for i, (a, a2) in enumerate(zip(x95_A, x95_A2)):
        delta = int(a - a2)
        ax.annotate(f"Δ={delta}", xy=(x[i] + w/2, a2),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, fontsize=9)
    ax.set_ylabel("Episodes to 95% asymptote")
    ax.set_title("Savings: A vs A2 across conditions\n(lower = faster, Δ = savings)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_bar_4cond.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_bar_4cond.png")

    # ── Thalamus routing (MoE conditions only) ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 3), sharey=True)
    for ax, (ci, cond) in zip(axes, [(2, CONDITIONS[2]), (3, CONDITIONS[3])]):
        curve  = curves[ci]
        phases = split_phases(curve["routing"])
        SW_R   = 20
        for pi, y_raw in enumerate(phases):
            y_sm = smooth(y_raw, SW_R)
            ep_x = ep_local[SW_R-1:]
            ax.plot(pi * N_PHASE_EP + ep_x, y_sm, color=COLORS[ci], lw=1.2)
        for pi in range(1, N_PHASES):
            ax.axvline(pi * N_PHASE_EP, color="gray", ls="--", lw=0.8)
        ax.set_ylim(-0.05, 1.1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["Expert 0", "0.5", "Expert 1"])
        ax.set_title(cond["name"])
        ax.set_xlabel("Episode")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Thalamus routing | separation = context-specific gating")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/thalamus_routing_4cond.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved thalamus_routing_4cond.png")

    # ── Savings overlay A vs A2 ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, N_CONDS, figsize=(5*N_CONDS, 4), sharey=True)
    SW_OV = 25
    for ax, cond, curve, color in zip(axes, CONDITIONS, curves, COLORS):
        angle_phases = split_phases(curve["first_angle"])
        for idx, ls, label in [(1, "-", "A"), (3, "--", "A2")]:
            y_raw = np.array(angle_phases[idx])
            y_sm  = smooth(y_raw, SW_OV)
            ep_sm = ep_local[SW_OV-1:]
            mask  = ep_sm <= 500
            ax.plot(ep_sm[mask], y_sm[mask], color=color, ls=ls, lw=2, label=label)
        ax.axhline(-45, color="gray", ls=":", lw=1, label="-45° target")
        ax.set_title(cond["name"], fontsize=9)
        ax.set_xlabel("Episode within phase")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("First-step angle (deg)")
    fig.suptitle("Savings overlay: A vs A2 | first 500 episodes")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_overlay_4cond.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_overlay_4cond.png")

    # ── savings_all_angle_overlay: N×2 (A vs A2, B vs B2) ────────────────────
    SW_OV2 = 25
    fig, axes = plt.subplots(N_CONDS, 2, figsize=(14, 4 * N_CONDS))
    for row, (cond, curve, color) in enumerate(zip(CONDITIONS, curves, COLORS)):
        angle_phases = split_phases(curve["first_angle"])
        for col, (idx1, idx2, target, title) in enumerate([
            (1, 3, -45, "A vs A2  (rot+45°)"),
            (2, 4, None, "B vs B2  (curl+)"),
        ]):
            ax = axes[row, col]
            for idx, shade, label in [(idx1, 0.4, PHASE_LABELS[idx1]),
                                       (idx2, 1.0, PHASE_LABELS[idx2])]:
                y_raw = np.array(angle_phases[idx])
                y_sm  = smooth(y_raw, SW_OV2)
                ep_sm = ep_local[SW_OV2 - 1:]
                mask  = ep_sm <= 350
                ax.plot(ep_sm[mask], y_sm[mask], color=color, lw=2,
                        alpha=shade, label=label)
            if target is not None:
                ax.axhline(target, color="gray", ls="--", lw=1.2,
                           label=f"{target:+d}° target")
            ax.set_xlim(1, 350)
            ax.set_ylim(-100, 50)
            ax.legend(fontsize=7, loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{cond['name']}\nAngle (deg)", fontsize=8, color=color)
            if row == N_CONDS - 1:
                ax.set_xlabel("Episode within phase", fontsize=8)
    fig.suptitle("ABA savings — angle overlay | faded=1st, solid=2nd exposure",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_all_angle_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_all_angle_overlay.png")

    # ── savings_all_acc_asymptote: N×3 (A, B, A2 phases) ─────────────────────
    ABA_INDICES = [1, 2, 3]
    ABA_LABELS  = [PHASE_LABELS[i] for i in ABA_INDICES]
    SW_ACC = 20

    fig, axes = plt.subplots(N_CONDS, 3, figsize=(15, 4 * N_CONDS), sharey=True)
    for row, (cond, curve, color) in enumerate(zip(CONDITIONS, curves, COLORS)):
        succ_all = split_phases(curve["success"])
        for col, (phase_idx, lbl) in enumerate(zip(ABA_INDICES, ABA_LABELS)):
            ax = axes[row, col]
            y_raw = np.array(succ_all[phase_idx])
            y_sm  = smooth(y_raw, SW_ACC)
            ep_sm = ep_local[SW_ACC - 1:]
            ax.plot(ep_sm, y_sm, color=color, lw=2, alpha=0.5)
            L, x95 = fit_x95(y_sm, ep_sm)
            if L is not None:
                y_fit = L + (y_sm[0] - L) * np.exp(
                    -(-np.log(0.05) / x95 if x95 else 1e-3) * ep_sm)
                ax.plot(ep_sm, y_fit, color="tomato", ls="--", lw=1.5,
                        label=f"asym={L:.2f}")
            if x95 is not None and x95 < N_PHASE_EP:
                ax.axvline(x95, color="black", ls=":", lw=1.2)
                ax.text(x95, 0.05, f"ep {int(x95)}", color="black",
                        fontsize=7, ha="center", va="bottom")
            ax.set_xlim(1, N_PHASE_EP)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(lbl, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{cond['name']}\nSuccess rate", fontsize=8,
                              color=color, fontweight="bold")
            if row == N_CONDS - 1:
                ax.set_xlabel("Episode within phase", fontsize=8)
    fig.suptitle("ABA accuracy with asymptote fits — 4 conditions\n"
                 "(rolling mean w=20 | red dashed = fit | dotted = x95)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_all_acc_asymptote.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_all_acc_asymptote.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
    data_path = f"{FIG_DIR}/data_4cond.npz"

    if os.path.exists(data_path) and "--replot" in sys.argv:
        print("Loading saved data — skipping training.")
        d      = np.load(data_path, allow_pickle=True)
        curves = list(d["curves"])
    else:
        curves = []
        for cond in CONDITIONS:
            print(f"\n=== {cond['name']} ===")
            mc = MotorCortex(use_cb_mc=cond["use_cb_mc"],
                             use_moe=cond["use_moe"]).to(device)

            if cond["use_moe"]:
                mc.thalamus.router.load_state_dict({
                    "weight": thal_ckpt["thalamus_state_dict"]["router.weight"],
                    "bias":   thal_ckpt["thalamus_state_dict"]["router.bias"]})
                for p in mc.thalamus.router.parameters():
                    p.requires_grad_(False)
                print(f"  Loaded + frozen pretrained thalamus.")

            print(f"  Pre-training experts ({N_PRETRAIN_EP} eps/ctx)...")
            pretrain_experts(ec, cb, mc, device, use_ec=cond["use_ec"], seed=42)

            curve = run_aba(ec, cb, mc, device, use_ec=cond["use_ec"], seed=0)
            curves.append(curve)

        np.savez(data_path, curves=curves)
        print(f"\nSaved data to {data_path}")

    plot_all(curves)
    print("Done.")


if __name__ == "__main__":
    main()

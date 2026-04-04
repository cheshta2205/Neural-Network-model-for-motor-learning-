"""
MC_savings_Thalamus_.py

ABA savings experiment with Thalamus-gated Mixture of Experts MC.

Architecture:
    EC (frozen) → z_ec → Thalamus → routes to MC expert k
    CB (frozen) → z_cb → MC expert k input [s_t, z_cb, g]
    z_ec does NOT enter MC expert directly — thalamus intercepts it.

Key idea:
    Thalamus learns which MC expert handles which context.
    When in context A: thalamus activates expert A, expert B receives zero gradient.
    When in context B: thalamus activates expert B, expert A receives zero gradient.
    → Structural protection against catastrophic forgetting — no explicit penalty needed.

Routing mechanism: Soft routing via softmax (biologically motivated)
    Forward  : soft weights → both experts always contribute to action output
    Backward : gradients flow to ALL experts proportional to their routing weight
               → no expert starved of gradient → no routing collapse
    Temperature annealed from TAU_START → TAU_END (sharpens weights over training).

Why soft routing instead of hard Gumbel:
    Hard routing causes collapse — one expert dominates, the other never learns.
    Soft routing: both experts train simultaneously, specialise via competition.
    Biologically motivated: thalamus modulates gain, does not binary-switch circuits.

EC+CB checkpoint: ectocb_8ctx_seq80_z32.pt  (pretrained, frozen throughout)
Phases: [pre: B, A, B, A2, B2]
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ── Hyperparameters ────────────────────────────────────────────────────────────
LATENT_DIM  = 32
EC_HIDDEN   = 256
CB_HIDDEN   = 256
EC_INPUT    = 8
CB_PROJ_DIM = 32
MC_HIDDEN   = 256
# MC expert input: [s_t(2), z_cb(32), g(2)] — z_ec removed, thalamus handles context
MC_INPUT    = 2 + CB_PROJ_DIM + 2   # 36

N_EXPERTS   = 2      # CTX_A=expert 0, CTX_B=expert 1 (matched to pretrain_thalamus.py)
TAU_START   = 1.0    # Gumbel temperature at start of training (softer routing)
TAU_END     = 0.1    # Gumbel temperature at end of training (sharper routing)
EMA_ALPHA   = 0.5    # EMA smoothing for thalamus routing (higher = more history)

MAX_STEPS    = 20
N_PHASE_EP   = 2000
GOAL_RADIUS  = 0.1
GRID_SIZE    = 4.0
DT           = 1.0
LR_MC        = 1e-3

CKPT         = "checkpoints/ectocb_8ctx_seq80_z32.pt"
CKPT_THALAMUS  = "checkpoints/thalamus_pretrained_rot45curl_persistent_ema05.pt"
FIG_DIR        = "../../figs/ECtoCB/MC_policy/MC_savings_ABA_hardtask/newtask_thalamus_rot45curl_persistent_ema05_pretrained"
N_PRETRAIN_EP  = 500   # episodes per context to warm-start both experts before ABA

# ── Phases ────────────────────────────────────────────────────────────────────
CTX_A  = (1.0,  np.pi/4, 0.0, 0.0, 0.0, 0.0, "A: rot+45")
CTX_B  = (1.0,  0.0,     1.0, 0.0, 0.0, 0.0, "B: curl+")
PHASES = [CTX_B, CTX_A, CTX_B, CTX_A, CTX_B]
PHASE_LABELS = ["pre: curl+", "A: rot+45°", "B: curl+", "A2: rot+45°", "B2: curl+"]


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
    """
    Single MC expert — a GRU with input [s_t, z_cb, g].
    z_ec is NOT an input here. Context enters only via thalamus routing.
    Each expert has completely independent weights.
    """
    def __init__(self):
        super().__init__()
        self.hidden_dim = MC_HIDDEN
        self.cell    = nn.GRUCell(MC_INPUT, MC_HIDDEN)
        self.readout = nn.Linear(MC_HIDDEN, 2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, g, h_mc, z_cb):
        """
        s_t  : (1, 2)
        g    : (1, 2)
        h_mc : (1, MC_HIDDEN)
        z_cb : (1, CB_PROJ_DIM)
        """
        inp   = torch.cat([s_t, z_cb, g], dim=-1)   # (1, 36)
        h_new = self.cell(inp, h_mc)
        return torch.tanh(self.readout(h_new)), h_new


class Thalamus(nn.Module):
    """
    Routes z_ec to MC experts via soft weighting.

    Uses softmax (not hard Gumbel-softmax):
        Forward : soft weights — both experts always contribute to action
        Backward: gradients flow to ALL experts proportional to their weight
                  → no expert is starved of gradient → no routing collapse

    When EC is lesioned (z_ec=0): logits=0 → uniform weights [0.5, 0.5] → balanced routing.
    """
    def __init__(self, latent_dim=LATENT_DIM, n_experts=N_EXPERTS):
        super().__init__()
        self.router = nn.Linear(latent_dim, n_experts)

    def forward(self, z_ec, temperature=1.0):
        logits = self.router(z_ec)                                        # (B, n_experts)
        return F.gumbel_softmax(logits, tau=temperature, hard=True)       # (B, n_experts) one-hot


class MotorCortexMoE(nn.Module):
    """
    Mixture-of-Experts MC with Thalamus gating.

    Step:
      1. Thalamus reads z_ec → soft routing weights (B, N_EXPERTS) via softmax
      2. All experts compute actions from [s_t, z_cb, g]
      3. Weighted sum → both experts always contribute to action
      4. Both experts always get gradient (proportional to their weight)
         → no routing collapse, natural specialisation over time

    Thalamus trains via task loss backprop through weighted sum.
    """
    def __init__(self, n_experts=N_EXPERTS):
        super().__init__()
        self.n_experts = n_experts
        self.thalamus  = Thalamus(LATENT_DIM, n_experts)
        self.experts   = nn.ModuleList([MCExpert() for _ in range(n_experts)])
        W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
        self.register_buffer("cb_proj", W / W.norm(dim=0, keepdim=True))

    def init_hidden(self, batch_size, device):
        return [expert.init_hidden(batch_size, device) for expert in self.experts]

    def step(self, s_t, z_ec, g, h_mc_list, cb_hidden, temperature=1.0):
        """
        s_t       : (1, 2)
        z_ec      : (1, LATENT_DIM) — goes to thalamus only, not experts
        g         : (1, 2)
        h_mc_list : list[N_EXPERTS] of (1, MC_HIDDEN)
        cb_hidden : (1, CB_HIDDEN)
        temperature: Gumbel temperature (annealed during training)

        Returns:
            action     : (1, 2)
            new_h_list : list[N_EXPERTS] updated hidden states
            weights    : (1, N_EXPERTS) soft routing weights
            expert_idx : float, weight of expert 1 (0=all expert0, 1=all expert1)
        """
        z_cb    = cb_hidden @ self.cb_proj                          # (1, CB_PROJ_DIM)
        weights = self.thalamus(z_ec, temperature=temperature)      # (1, N_EXPERTS) one-hot

        # Run all experts
        new_h_list = []
        actions    = []
        for i, expert in enumerate(self.experts):
            a_i, h_i = expert.step(s_t, g, h_mc_list[i], z_cb)
            actions.append(a_i)        # (1, 2)
            new_h_list.append(h_i)

        # Weighted sum: one-hot → only selected expert contributes
        # Non-selected experts have weight=0 → zero gradient through action path
        actions_stack = torch.stack(actions, dim=1)     # (1, N_EXPERTS, 2)
        w             = weights.unsqueeze(-1)            # (1, N_EXPERTS, 1)
        action        = (actions_stack * w).sum(dim=1)   # (1, 2)

        expert_idx = weights[0, 1].item()   # fraction routed to expert 1 (0→1 scale)

        return action, new_h_list, weights, expert_idx


# ── Differentiable env step ───────────────────────────────────────────────────
def env_step_torch(pos, vel, action, beta, theta, gamma, g, bx, by, device):
    cos_t = torch.tensor(float(np.cos(theta)), dtype=torch.float32, device=device)
    sin_t = torch.tensor(float(np.sin(theta)), dtype=torch.float32, device=device)
    R     = torch.stack([torch.stack([cos_t, -sin_t]),
                         torch.stack([sin_t,  cos_t])])
    curl  = torch.tensor([[0., -1.], [1., 0.]], device=device)
    b_vec = torch.tensor([float(bx), float(by)], device=device)
    F_     = beta * (R @ action) + gamma * (curl @ vel) + g * b_vec
    vel_new = vel + F_
    pos_new = torch.clamp(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new.detach()


# ── Single episode ─────────────────────────────────────────────────────────────
def run_episode(ec, cb, mc, device, ctx, rng, train=True, ec_lesion=False, temperature=1.0,
                h_ec_in=None, z_smooth_in=None):
    """
    h_ec_in, z_smooth_in: carry-over EC state from previous episode (persistent EC).
    If None, initialises from zeros (start of a new phase).
    Returns (loss, success, steps, dist_norm, mean_angle, first_angle, mean_routing,
             h_ec_out, z_smooth_out) so caller can persist EC state across episodes.
    """
    beta, theta, gamma, g, bx, by, _ = ctx

    start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(start - goal) < 0.5:
        goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    g_t       = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos       = torch.tensor(start, dtype=torch.float32, device=device)
    vel       = torch.zeros(2, device=device)
    h_ec      = h_ec_in     if h_ec_in     is not None else ec.init_hidden(1, device)
    h_mc_list = mc.init_hidden(1, device)
    z_t       = torch.zeros(1, LATENT_DIM, device=device)
    z_smooth  = z_smooth_in if z_smooth_in is not None else torch.zeros(1, LATENT_DIM, device=device)
    prev_a    = torch.zeros(1, 2, device=device)

    step_losses  = []
    dist_seq     = []
    angle_seq    = []
    routing_seq  = []
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

        # EMA smooth z_ec before routing — accumulates context signal across steps
        with torch.no_grad():
            z_smooth = EMA_ALPHA * z_smooth + (1 - EMA_ALPHA) * z_t

        # z_smooth goes to thalamus; if lesioned z_ec=0 → thalamus routes randomly
        z_ec_in = torch.zeros(1, LATENT_DIM, device=device) if ec_lesion else z_smooth

        a_t, h_mc_list, weights, expert_idx = mc.step(
            pos.unsqueeze(0), z_ec_in, g_t.unsqueeze(0), h_mc_list, cb_h,
            temperature=temperature)
        routing_seq.append(expert_idx)

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
    mean_angle  = np.mean(angle_seq)  if angle_seq  else 0.0
    first_angle = first_angle if first_angle is not None else 0.0
    # fraction of steps routed to expert 1 (0.0=always expert 0, 1.0=always expert 1)
    mean_routing = np.mean(routing_seq) if routing_seq else 0.0

    loss = torch.stack(step_losses).sum() if step_losses else \
           torch.tensor(0.0, device=device, requires_grad=True)
    return loss, float(success), steps, dist_norm, mean_angle, first_angle, mean_routing, \
           h_ec.detach(), z_smooth.detach()


# ── Expert pre-training ────────────────────────────────────────────────────────
def pretrain_experts(ec, cb, mc, device, seed=42):
    """
    Warm-start both experts before ABA begins.
    Runs N_PRETRAIN_EP episodes for each context with routing fixed at 0.5
    so both experts receive equal gradient and learn basic motor control.
    Router stays frozen throughout.
    """
    ec.eval(); cb.eval(); mc.train()
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.Adam(
        [p for p in mc.parameters() if p.requires_grad], lr=LR_MC)
    rng = np.random.default_rng(seed)

    for ctx in [CTX_A, CTX_B]:
        label = ctx[-1]
        print(f"  Pre-training experts on {label} ({N_PRETRAIN_EP} episodes)...")
        for _ in range(N_PRETRAIN_EP):
            optimizer.zero_grad()
            # temperature=100 → Gumbel softmax ≈ uniform → both experts get ~equal gradient
            loss, _, _, _, _, _, _, _, _ = run_episode(
                ec, cb, mc, device, ctx, rng,
                train=True, ec_lesion=False, temperature=100.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            optimizer.step()
    print("  Expert pre-training done.\n")


# ── ABA training loop ──────────────────────────────────────────────────────────
def run_aba(ec, cb, mc, device, seed=0, ec_lesion=False):
    ec.eval(); cb.eval(); mc.train()
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)

    optimizer     = torch.optim.Adam(mc.parameters(), lr=LR_MC)
    rng           = np.random.default_rng(seed)
    total_eps     = N_PHASE_EP * len(PHASES)

    curve = {"ep": [], "success": [], "steps": [], "dist_norm": [],
             "angle": [], "first_angle": [], "phase": [], "routing": []}
    ep_global = 0

    for phase_idx, ctx in enumerate(PHASES):
        label = PHASE_LABELS[phase_idx]
        print(f"\nPhase {label} ({N_PHASE_EP} episodes)...")

        # Reset EC state at phase boundary — context has changed
        h_ec_carry     = None
        z_smooth_carry = None

        for ep in range(1, N_PHASE_EP + 1):
            ep_global += 1
            tau = TAU_START - (TAU_START - TAU_END) * (ep_global / total_eps)
            mc.train()

            optimizer.zero_grad()
            loss, _, _, _, _, _, _, h_ec_carry, z_smooth_carry = run_episode(
                ec, cb, mc, device, ctx, rng,
                train=True, ec_lesion=ec_lesion, temperature=tau,
                h_ec_in=h_ec_carry, z_smooth_in=z_smooth_carry)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            optimizer.step()

            mc.eval()
            with torch.no_grad():
                _, succ, steps, dist_norm, mean_angle, first_angle, routing, _, _ = run_episode(
                    ec, cb, mc, device, ctx, rng,
                    train=False, ec_lesion=ec_lesion, temperature=tau,
                    h_ec_in=h_ec_carry, z_smooth_in=z_smooth_carry)

            curve["ep"].append(ep_global)
            curve["success"].append(float(succ))
            curve["steps"].append(steps)
            curve["dist_norm"].append(dist_norm)
            curve["angle"].append(mean_angle)
            curve["first_angle"].append(first_angle)
            curve["phase"].append(phase_idx)
            curve["routing"].append(routing)

            if ep % 500 == 0:
                recent_acc     = np.mean(curve["success"][-100:])
                recent_angle   = np.mean(curve["angle"][-100:])
                recent_routing = np.mean(curve["routing"][-100:])
                print(f"  [{label}] ep {ep:4d} | acc={recent_acc:.2f} | "
                      f"angle={recent_angle:.1f}° | routing={recent_routing:.2f} | τ={tau:.3f}")

    return curve


# ── Plots ──────────────────────────────────────────────────────────────────────
def smooth(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fit_asymptote(x, y):
    from scipy.optimize import curve_fit
    def model(x, L, y0, k):
        return L + (y0 - L) * np.exp(-k * x)
    try:
        p0 = [float(y[-1]), float(y[0]), 1e-3]
        popt, _ = curve_fit(model, x, y, p0=p0, maxfev=5000)
        L, y0, k = popt
        x_95 = -np.log(0.05) / k if k > 0 and abs(L - y0) > 1e-6 else None
        return L, y0, k, x_95
    except Exception:
        return None, None, None, None


def plot_results(curve_intact, curve_lesion):
    os.makedirs(FIG_DIR, exist_ok=True)

    SW       = 20
    N_PHASES = len(PHASES)

    def split_phases(arr):
        arr = np.array(arr)
        return [arr[i * N_PHASE_EP:(i + 1) * N_PHASE_EP] for i in range(N_PHASES)]

    succ_phases  = split_phases(curve_intact["success"])
    steps_phases = split_phases(curve_intact["steps"])
    dist_phases  = split_phases(curve_intact["dist_norm"])
    ep_local     = np.arange(1, N_PHASE_EP + 1)
    phase_boundaries = [i * N_PHASE_EP for i in range(1, N_PHASES)]

    # ── Learning curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 4))
    for ax, phases, use_smooth, ylabel in zip(
        axes,
        [succ_phases, steps_phases, dist_phases],
        [True, True, False],
        ["Success rate", "Mean steps to goal", "Mean norm. dist to goal"],
    ):
        for phase_idx, y_raw in enumerate(phases):
            y    = smooth(y_raw, SW) if use_smooth else y_raw
            ep_x = ep_local[SW - 1:] if use_smooth else ep_local
            xoff = phase_idx * N_PHASE_EP
            ax.plot(xoff + ep_x, y, color="steelblue", lw=2)

            L, y0, k, x_95 = fit_asymptote(ep_x, y)
            print(f"  {ylabel} | {PHASE_LABELS[phase_idx]}: asymptote={L:.3f}, "
                  f"ep_95={int(x_95) if x_95 else 'N/A'}")
            y_min, y_max = np.min(y), np.max(y)
            y_range = y_max - y_min if y_max != y_min else 1.0
            if L is not None and y_min - 2*y_range < L < y_max + 2*y_range:
                ax.axhline(L, xmin=xoff / (N_PHASES * N_PHASE_EP),
                           xmax=(xoff + N_PHASE_EP) / (N_PHASES * N_PHASE_EP),
                           color="tomato", ls="--", lw=1.2)
            if x_95 is not None and x_95 < N_PHASE_EP:
                ax.axvline(xoff + x_95, color="tomato", ls=":", lw=1.2)

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

    fig.suptitle("ABAB savings | Thalamus MoE gating\n"
                 "A=rot+45°, B=curl+  (red=asymptote fit)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/aba_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Action angle ─────────────────────────────────────────────────────────
    angle_phases = split_phases(curve_intact["first_angle"])

    fig, ax = plt.subplots(figsize=(16, 4))
    for phase_idx, y_raw in enumerate(angle_phases):
        xoff = phase_idx * N_PHASE_EP
        ax.plot(xoff + ep_local, np.array(y_raw), color="steelblue", lw=0.8, alpha=0.7)
    ax.axhline(-45, color="tomato",   ls="--", lw=1.2, label="-45° (target A/A2: rot+45°)")
    # curl+ has no fixed angle target (velocity-dependent force)
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
        ax.text(mid, 170, lbl, ha="center", va="bottom", fontsize=8, color="dimgray",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))
    ax.set_xlabel("Episode within phase")
    ax.set_ylabel("First-step action angle (degrees)")
    ax.set_title("First-step action angle | Thalamus MoE | rot+45° vs curl+")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/aba_action_angle.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Savings overlay ───────────────────────────────────────────────────────
    SW_OV = 25
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (idx1, idx2), target, title in zip(
        axes,
        [(1, 3), (2, 4)],
        [-45, None],
        ["A vs A2  (rot+45°)", "B vs B2  (curl+)"],
    ):
        for idx, color, label in [(idx1, "steelblue", PHASE_LABELS[idx1]),
                                   (idx2, "tomato",    PHASE_LABELS[idx2])]:
            y_raw = np.array(angle_phases[idx])
            y_sm  = smooth(y_raw, SW_OV)
            ep_sm = ep_local[SW_OV - 1:]
            mask  = ep_sm <= 350
            ax.plot(ep_sm[mask], y_sm[mask], color=color, lw=2, label=label)
        if target is not None:
            ax.axhline(target, color="gray", ls="--", lw=1.2, label=f"{target:+d}° (target)")
        ax.set_xlabel("Episode within phase")
        ax.set_ylabel("First-step angle (deg)")
        ax.set_title(title)
        ax.set_xlim(1, 350)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Savings overlay | Thalamus MoE | rot+45° vs curl+  (rolling mean w={SW_OV}, ep 1-350)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/aba_savings_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Thalamus routing per phase ────────────────────────────────────────────
    # Shows which expert thalamus selected over time — key diagnostic for this model.
    # Correct behaviour: A phases → expert 0 (or 1) consistently, B phases → the other.
    routing_phases = split_phases(curve_intact["routing"])
    SW_R = 20

    fig, ax = plt.subplots(figsize=(16, 3))
    for phase_idx, y_raw in enumerate(routing_phases):
        xoff = phase_idx * N_PHASE_EP
        y_sm = smooth(y_raw, SW_R)
        ep_x = ep_local[SW_R - 1:]
        ax.plot(xoff + ep_x, y_sm, color="steelblue", lw=1.5)

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
        ax.text(mid, 1.05, lbl, ha="center", va="bottom", fontsize=8, color="dimgray",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))

    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["Expert 0", "0.5", "Expert 1"])
    ax.set_xlabel("Episode within phase")
    ax.set_ylabel("Mean routing (fraction → expert 1)")
    ax.set_title(f"Thalamus routing per phase | rolling mean w={SW_R}\n"
                 "Separation between A and B phases = thalamus learned context-specific routing")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/thalamus_routing.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Figures saved to {FIG_DIR}/")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"N_EXPERTS={N_EXPERTS} | TAU {TAU_START}→{TAU_END} | "
          f"MC_INPUT={MC_INPUT} (no z_ec) | LR={LR_MC}")

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
        # Load pretrained thalamus then FREEZE router — prevents collapse from task loss
        thal_ckpt = torch.load(CKPT_THALAMUS, map_location=device)
        mc_intact = MotorCortexMoE().to(device)
        mc_lesion = MotorCortexMoE().to(device)
        for mc in [mc_intact, mc_lesion]:
            mc.thalamus.router.load_state_dict({
                "weight": thal_ckpt["thalamus_state_dict"]["router.weight"],
                "bias":   thal_ckpt["thalamus_state_dict"]["router.bias"]})
            for p in mc.thalamus.router.parameters():
                p.requires_grad_(False)   # freeze — routing cannot drift during MC training
        print(f"Loaded + FROZEN pretrained thalamus from {CKPT_THALAMUS}")

        print("\n=== Pre-training experts (warm-start both before ABA) ===")
        pretrain_experts(ec, cb, mc_intact, device, seed=42)
        pretrain_experts(ec, cb, mc_lesion, device, seed=42)

        print("\n=== EC INTACT + Thalamus MoE ===")
        curve_intact = run_aba(ec, cb, mc_intact, device, seed=0, ec_lesion=False)

        print("\n=== EC LESION + Thalamus MoE ===")
        curve_lesion = run_aba(ec, cb, mc_lesion, device, seed=0, ec_lesion=True)

        torch.save(mc_intact.state_dict(), f"{FIG_DIR}/mc_intact.pt")
        torch.save(mc_lesion.state_dict(), f"{FIG_DIR}/mc_lesion.pt")
        np.savez(data_path, curve_intact=curve_intact, curve_lesion=curve_lesion)
        print(f"\nSaved checkpoints and data to {FIG_DIR}/")

    plot_results(curve_intact, curve_lesion)
    print("Done.")


if __name__ == "__main__":
    main()

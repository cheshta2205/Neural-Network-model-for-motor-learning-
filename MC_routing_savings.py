"""
MC_routing_savings.py

ABA savings experiment — same 4 conditions as MC_norouting_savings but with
thalamic routing added for EC conditions.

4 conditions (mirroring MC_norouting_savings):
  1. MC only     — no EC, no CB. 2 experts, equal 0.5/0.5 weighting (no routing).
  2. MC + CB     — no EC, CB hidden to experts. Equal weighting (no routing).
  3. MC + EC     — EC present → z_ec goes to frozen thalamus → hard-routes to one expert.
                   Experts see [s, g] only. EC never feeds experts directly.
  4. MC + EC+CB  — EC routes via thalamus + experts also get z_cb from CB.

Key hypothesis:
  Routing (conds 3 & 4) protects expert-A weights during B training → savings in A2.
  Without routing (conds 1 & 2), both experts get equal gradient on every episode
  → both overwritten during B → no savings. Same as MC_norouting_savings result.

All conditions use 2 experts (EXPERT_HIDDEN=128 each ≈ same capacity as single
256-dim GRU). Routing is the only structural difference.

EC state persists across ALL episodes (no reset at phase boundaries).
Thalamus pretrained and frozen throughout MC training.

Contexts (in-distribution for pretrained EC+thalamus):
  CTX_A: rot+60° (theta=π/3)
  CTX_B: curl+   (gamma=1.0)

Checkpoints:
  EC+CB : checkpoints/ectocb_8ctx_seq80_z32.pt
  Thal  : checkpoints/thalamus_pretrained_rot60curl.pt
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ── Hyperparameters ────────────────────────────────────────────────────────────
LATENT_DIM    = 32
EC_HIDDEN     = 256
CB_HIDDEN     = 256
EC_INPUT      = 8
CB_PROJ_DIM   = 32
N_EXPERTS     = 2
EXPERT_HIDDEN = 128   # half of 256 → 2 experts ≈ same capacity as single 256-dim GRU
EMA_ALPHA     = 0.5   # must match pretrain_thalamus.py
TAU_START     = 1.0
TAU_END       = 0.1

MAX_STEPS     = 20
N_PHASE_EP    = 2000
GOAL_RADIUS   = 0.1
GRID_SIZE     = 4.0
DT            = 1.0
LR_MC         = 1e-3
N_PRETRAIN_EP = 500   # warm-start each expert before ABA

CKPT_ECTOCB  = "checkpoints/ectocb_8ctx_seq80_z32.pt"
CKPT_THAL    = "checkpoints/thalamus_pretrained_rot60curl.pt"
FIG_DIR      = "../../figs/ECtoCB/MC_EC_CB/MC_savings_routing_60curl60_nopretrain"

# ── Contexts ───────────────────────────────────────────────────────────────────
CTX_A = (1.0, np.pi/3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B = (1.0, 0.0,     1.0, 0.0, 0.0, 0.0, "B: curl+")

PHASES       = [CTX_B, CTX_A, CTX_B, CTX_A, CTX_B]
PHASE_LABELS = ["pre: curl+", "A: rot+60°", "B: curl+", "A2: rot+60°", "B2: curl+"]

# ── Conditions ─────────────────────────────────────────────────────────────────
# use_ec      : whether EC runs and produces z_ec
# use_cb      : whether CB hidden (z_cb) is fed to experts
# use_routing : True  → z_ec → thalamus → hard-select one expert
#               False → equal 0.5/0.5 weighting (no thalamus, no EC signal to MC)
CONDITIONS = [
    {"name": "MC only",    "use_ec": False, "use_cb": False, "use_routing": False, "retrain_cb": False},
    {"name": "MC + CB",    "use_ec": False, "use_cb": True,  "use_routing": False, "retrain_cb": True},
    {"name": "MC + EC",    "use_ec": True,  "use_cb": False, "use_routing": True,  "retrain_cb": False},
    {"name": "MC + EC+CB", "use_ec": True,  "use_cb": True,  "use_routing": True,  "retrain_cb": False},
]
COND_COLORS = ["dimgray", "steelblue", "seagreen", "tomato"]


# ── Models ─────────────────────────────────────────────────────────────────────
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
        return self.out(self.layer2(self.layer1(torch.cat([s, a, z], dim=-1))))

    def hidden(self, s, a, z):
        return self.layer2(self.layer1(torch.cat([s, a, z], dim=-1)))


class CerebellumCB_NoEC(nn.Module):
    """CB retrained without EC input — takes [s, a] only. Used in MC+CB condition."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(4, CB_HIDDEN), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(CB_HIDDEN, CB_HIDDEN), nn.ReLU())
        self.out    = nn.Linear(CB_HIDDEN, 2)

    def forward(self, s, a):
        return self.out(self.layer2(self.layer1(torch.cat([s, a], dim=-1))))

    def hidden(self, s, a):
        return self.layer2(self.layer1(torch.cat([s, a], dim=-1)))


class Thalamus(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(LATENT_DIM, N_EXPERTS)

    def forward(self, z_ec, temperature=1.0):
        return F.gumbel_softmax(self.router(z_ec), tau=temperature, hard=True)


class MCExpert(nn.Module):
    """Single GRU expert. Sees [s, g] always; optionally [s, z_cb, g]."""
    def __init__(self, use_cb=False):
        super().__init__()
        input_dim = 2 + 2  # s + g
        if use_cb:
            input_dim += CB_PROJ_DIM
        self.cell    = nn.GRUCell(input_dim, EXPERT_HIDDEN)
        self.readout = nn.Linear(EXPERT_HIDDEN, 2)
        self.use_cb  = use_cb

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, EXPERT_HIDDEN, device=device)

    def step(self, s_t, g, h, z_cb=None):
        parts = [s_t, g]
        if self.use_cb:
            parts.append(z_cb)
        h_new = self.cell(torch.cat(parts, dim=-1), h)
        return torch.tanh(self.readout(h_new)), h_new


class MotorCortexMoE(nn.Module):
    """
    2-expert MC. Used for all 4 conditions.

    use_routing=False: equal [0.5, 0.5] weighting — both experts always contribute,
                       both always get gradient. No context-selective protection.
    use_routing=True : thalamus reads z_ec → hard one-hot routing.
                       Only selected expert gets gradient → other expert protected.
    """
    def __init__(self, use_cb=False, use_routing=False):
        super().__init__()
        self.use_routing = use_routing
        self.use_cb      = use_cb
        self.experts     = nn.ModuleList([MCExpert(use_cb) for _ in range(N_EXPERTS)])
        self.thalamus    = Thalamus() if use_routing else None
        if use_cb:
            W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
            self.register_buffer("cb_proj", W / W.norm(dim=0, keepdim=True))

    def init_hidden(self, batch_size, device):
        return [e.init_hidden(batch_size, device) for e in self.experts]

    def step(self, s_t, z_ec, g, h_list, cb_hidden, temperature=1.0):
        z_cb = (cb_hidden @ self.cb_proj) if self.use_cb else None

        if self.use_routing:
            weights = self.thalamus(z_ec, temperature=temperature)  # one-hot (1, N_EXPERTS)
        else:
            weights = torch.full((1, N_EXPERTS), 1.0 / N_EXPERTS,
                                 device=s_t.device)  # equal weighting

        new_h, actions = [], []
        for i, expert in enumerate(self.experts):
            a_i, h_i = expert.step(s_t, g, h_list[i], z_cb)
            actions.append(a_i)
            new_h.append(h_i)

        action = (torch.stack(actions, dim=1) * weights.unsqueeze(-1)).sum(dim=1)
        return action, new_h


# ── Environment ────────────────────────────────────────────────────────────────
def env_step_torch(pos, vel, action, beta, theta, gamma, g, bx, by, device):
    cos_t = torch.tensor(float(np.cos(theta)), dtype=torch.float32, device=device)
    sin_t = torch.tensor(float(np.sin(theta)), dtype=torch.float32, device=device)
    R     = torch.stack([torch.stack([cos_t, -sin_t]),
                         torch.stack([sin_t,  cos_t])])
    curl  = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32, device=device)
    b_vec = torch.tensor([float(bx), float(by)], dtype=torch.float32, device=device)
    F_    = beta * (R @ action) + gamma * (curl @ vel) + g * b_vec
    vel_new = vel + F_
    pos_new = torch.clamp(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new.detach()


# ── Single episode ─────────────────────────────────────────────────────────────
def run_episode(ec, cb, mc, device, ctx, rng, train=True,
                use_ec=False, cb_noec=None,
                h_ec_in=None, z_t_in=None, z_smooth_in=None,
                temperature=1.0):
    beta, theta, gamma, g, bx, by, _ = ctx

    start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(start - goal) < 0.5:
        goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    g_t   = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos   = torch.tensor(start, dtype=torch.float32, device=device)
    vel   = torch.zeros(2, dtype=torch.float32, device=device)

    h_ec     = h_ec_in     if h_ec_in     is not None else ec.init_hidden(1, device)
    h_mc     = mc.init_hidden(1, device)
    z_t      = z_t_in      if z_t_in      is not None else torch.zeros(1, LATENT_DIM, device=device)
    z_smooth = z_smooth_in if z_smooth_in is not None else torch.zeros(1, LATENT_DIM, device=device)
    prev_a   = torch.zeros(1, 2, device=device)

    step_losses, dist_seq, angle_seq = [], [], []
    success     = False
    steps       = 0
    first_angle = None
    init_dist   = torch.norm(pos - g_t).item()

    for t in range(MAX_STEPS):
        d = torch.norm(pos - g_t).item()
        dist_seq.append(d)
        if d < GOAL_RADIUS:
            success = True; break

        with torch.no_grad():
            cb_h = (cb_noec.hidden(pos.unsqueeze(0), prev_a)
                    if cb_noec is not None
                    else cb.hidden(pos.unsqueeze(0), prev_a, z_t))
            z_smooth = EMA_ALPHA * z_smooth + (1 - EMA_ALPHA) * z_t

        z_ec_in = z_smooth if use_ec else torch.zeros(1, LATENT_DIM, device=device)

        a_t, h_mc = mc.step(pos.unsqueeze(0), z_ec_in, g_t.unsqueeze(0),
                             h_mc, cb_h, temperature=temperature)

        s_next, vel_next = env_step_torch(
            pos, vel, a_t.squeeze(0), beta, theta, gamma, g, bx, by, device)

        if train:
            step_losses.append(torch.sum((g_t - s_next) ** 2))
            if cb_noec is not None:
                pred = cb_noec(pos.unsqueeze(0), a_t.detach())
                step_losses.append(torch.sum((s_next.detach().unsqueeze(0) - pred) ** 2))

        with torch.no_grad():
            a_np  = a_t.squeeze(0).detach().cpu().numpy()
            gd    = (g_t - pos).cpu().numpy()
            alpha = np.degrees(np.arctan2(
                gd[0]*a_np[1] - gd[1]*a_np[0],
                a_np[0]*gd[0] + a_np[1]*gd[1]))
            angle_seq.append(alpha)
            if t == 0:
                first_angle = alpha

        with torch.no_grad():
            if use_ec:
                s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                eps   = s_next.detach().unsqueeze(0) - cb(pos.unsqueeze(0), a_t.detach(), z_t)
                z_t, h_ec = ec.step(
                    torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

        prev_a = a_t.detach()
        pos    = s_next.detach()
        vel    = vel_next
        steps += 1

    dist_norm   = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
    mean_angle  = float(np.mean(angle_seq)) if angle_seq else 0.0
    first_angle = first_angle if first_angle is not None else 0.0

    loss = (torch.stack(step_losses).sum() if step_losses
            else torch.tensor(0.0, device=device, requires_grad=True))

    return loss, float(success), steps, dist_norm, mean_angle, first_angle, \
           h_ec.detach(), z_t.detach(), z_smooth.detach()


# ── Expert warm-start (routing conditions only) ────────────────────────────────
def pretrain_experts(ec, cb, mc, device, use_ec, cb_noec=None, seed=0):
    """Warm-start both experts with temperature≈∞ so both get equal gradient."""
    ec.eval(); cb.eval(); mc.train()
    if cb_noec is not None: cb_noec.train()
    params = list(mc.parameters())
    if cb_noec is not None: params += list(cb_noec.parameters())
    optimizer = torch.optim.Adam(
        [p for p in params if p.requires_grad], lr=LR_MC)
    rng = np.random.default_rng(seed)
    for ctx in [CTX_A, CTX_B]:
        print(f"    Pretraining experts on {ctx[-1]} ({N_PRETRAIN_EP} ep)...")
        h_ec_carry  = ec.init_hidden(1, device)
        z_carry     = torch.zeros(1, LATENT_DIM, device=device)
        z_sm_carry  = torch.zeros(1, LATENT_DIM, device=device)
        for _ in range(N_PRETRAIN_EP):
            optimizer.zero_grad()
            loss, _, _, _, _, _, h_ec_carry, z_carry, z_sm_carry = run_episode(
                ec, cb, mc, device, ctx, rng,
                train=True, use_ec=use_ec, cb_noec=cb_noec,
                h_ec_in=h_ec_carry if use_ec else None,
                z_t_in=z_carry     if use_ec else None,
                z_smooth_in=z_sm_carry if use_ec else None,
                temperature=100.0)   # uniform routing → both experts learn
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()


# ── Training loop ──────────────────────────────────────────────────────────────
def run_aba(ec, cb, mc, device, seed=0, use_ec=False, cb_noec=None):
    ec.eval(); cb.eval()
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)
    if mc.thalamus is not None:
        for p in mc.thalamus.parameters(): p.requires_grad_(False)

    mc.train()
    if cb_noec is not None: cb_noec.train()
    params    = list(mc.parameters())
    if cb_noec is not None: params += list(cb_noec.parameters())
    optimizer  = torch.optim.Adam(
        [p for p in params if p.requires_grad], lr=LR_MC)
    rng        = np.random.default_rng(seed)
    total_eps  = N_PHASE_EP * len(PHASES)

    curve = {"ep": [], "success": [], "steps": [], "dist_norm": [],
             "angle": [], "first_angle": [], "phase": [],
             "probe_a_ep": [], "probe_a_succ": [], "probe_a_phase": []}

    ep_global   = 0
    h_ec_carry  = ec.init_hidden(1, device)
    z_carry     = torch.zeros(1, LATENT_DIM, device=device)
    z_sm_carry  = torch.zeros(1, LATENT_DIM, device=device)

    for phase_idx, ctx in enumerate(PHASES):
        label = PHASE_LABELS[phase_idx]
        print(f"\n  Phase {label} ({N_PHASE_EP} episodes)...")

        for ep in range(1, N_PHASE_EP + 1):
            ep_global += 1
            tau = TAU_START - (TAU_START - TAU_END) * (ep_global / total_eps)

            mc.train()
            optimizer.zero_grad()
            loss, _, _, _, _, _, h_ec_carry, z_carry, z_sm_carry = run_episode(
                ec, cb, mc, device, ctx, rng,
                train=True, use_ec=use_ec, cb_noec=cb_noec,
                h_ec_in=h_ec_carry if use_ec else None,
                z_t_in=z_carry     if use_ec else None,
                z_smooth_in=z_sm_carry if use_ec else None,
                temperature=tau)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            mc.eval()
            with torch.no_grad():
                _, succ, steps, dist_norm, mean_angle, first_angle, _, _, _ = run_episode(
                    ec, cb, mc, device, ctx, rng,
                    train=False, use_ec=use_ec, cb_noec=cb_noec,
                    h_ec_in=h_ec_carry if use_ec else None,
                    z_t_in=z_carry     if use_ec else None,
                    z_smooth_in=z_sm_carry if use_ec else None,
                    temperature=tau)

            curve["ep"].append(ep_global)
            curve["success"].append(float(succ))
            curve["steps"].append(float(steps))
            curve["dist_norm"].append(float(dist_norm))
            curve["angle"].append(mean_angle)
            curve["first_angle"].append(first_angle)
            curve["phase"].append(phase_idx)

            # Probe context-A performance during B phases
            if phase_idx in [2, 4]:
                with torch.no_grad():
                    _, probe_succ, _, _, _, _, _, _, _ = run_episode(
                        ec, cb, mc, device, CTX_A, rng,
                        train=False, use_ec=use_ec, cb_noec=cb_noec,
                        h_ec_in=h_ec_carry if use_ec else None,
                        z_t_in=z_carry     if use_ec else None,
                        z_smooth_in=z_sm_carry if use_ec else None,
                        temperature=tau)
                curve["probe_a_ep"].append(ep_global)
                curve["probe_a_succ"].append(float(probe_succ))
                curve["probe_a_phase"].append(phase_idx)

            if ep % 500 == 0:
                print(f"    ep {ep:4d} | acc(last100)={np.mean(curve['success'][-100:]):.2f} "
                      f"| angle(last100)={np.mean(curve['angle'][-100:]):.1f}°")

    return curve


# ── Plotting ───────────────────────────────────────────────────────────────────
def smooth(x, w):
    if len(x) < w: return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fit_asymptote(x, y):
    from scipy.optimize import curve_fit
    def model(xv, L, y0, k):
        return L + (y0 - L) * np.exp(-k * xv)
    try:
        popt, _ = curve_fit(model, x, y, p0=[float(y[-1]), float(y[0]), 1e-3], maxfev=5000)
        L, y0, k = popt
        x95 = -np.log(0.05) / k if k > 0 and abs(L - y0) > 1e-6 else None
        return L, y0, k, x95
    except Exception:
        return None, None, None, None


def split_phases(arr):
    arr = np.array(arr)
    return [arr[i * N_PHASE_EP:(i + 1) * N_PHASE_EP] for i in range(len(PHASES))]


def plot_all(curves_list, cond_names, colors):
    os.makedirs(FIG_DIR, exist_ok=True)

    N_CONDS  = len(CONDITIONS)
    ep_local = np.arange(1, N_PHASE_EP + 1)

    def _split(arr):
        return split_phases(arr)

    phase_boundaries = [i * N_PHASE_EP for i in range(1, len(PHASES))]

    # ── Fig 1: learning curves (success / steps / dist_norm) ──────────────────
    fig1, axes1 = plt.subplots(N_CONDS, 3, figsize=(22, 4 * N_CONDS), sharey="col")
    SW_SUCC = SW_STEPS = SW_DIST = 20
    metrics = [
        ("success",   "Success rate",            SW_SUCC,  True),
        ("steps",     "Mean steps to goal",      SW_STEPS, True),
        ("dist_norm", "Mean norm. dist to goal", SW_DIST,  True),
    ]
    for row, (cond, color, curve) in enumerate(zip(CONDITIONS, colors, curves_list)):
        for col, (metric, ylabel, sw, do_smooth) in enumerate(metrics):
            ax = axes1[row, col]
            phases = _split(curve[metric])
            for phase_idx, y_raw in enumerate(phases):
                y_raw = np.array(y_raw)
                y    = smooth(y_raw, sw) if do_smooth else y_raw
                ep_x = ep_local[sw - 1:] if (do_smooth and len(y_raw) >= sw) else ep_local[:len(y)]
                xoff = phase_idx * N_PHASE_EP
                ax.plot(xoff + ep_x, y, color=color, lw=1.6, alpha=0.9)
                if metric == "success" and len(y) > 10:
                    L, y0, k, x95 = fit_asymptote(ep_x, y)
                    if L is not None:
                        ax.plot(xoff + ep_x, L + (y0 - L) * np.exp(-k * ep_x),
                                color="black", ls="--", lw=1.0, alpha=0.8)
            for xb in phase_boundaries:
                ax.axvline(xb, color="gray", ls="--", lw=0.8)
            tick_pos, tick_lbl = [], []
            for i in range(len(PHASES)):
                for t in [0, 1000, 2000]:
                    tick_pos.append(i * N_PHASE_EP + t); tick_lbl.append(str(t))
            ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lbl, fontsize=6)
            if row == 0: ax.set_title(ylabel, fontsize=10)
            if col == 0: ax.set_ylabel(f"{cond['name']}\n{ylabel}", fontsize=8)
            if row == N_CONDS - 1: ax.set_xlabel("Episode within phase", fontsize=8)
            if metric == "success": ax.set_ylim(0, 1.05)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                y_top = ax.get_ylim()[1]
                for i, lbl in enumerate(PHASE_LABELS):
                    mid = i * N_PHASE_EP + N_PHASE_EP // 2
                    ax.text(mid, y_top, lbl, ha="center", va="bottom", fontsize=6,
                            color="dimgray",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
    fig1.suptitle(
        "ABA savings — 4 conditions, with routing/gating\n"
        "(rolling means; black dashed on success = exponential fit)",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_4cond_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_4cond_learning_curves.png")

    # ── Fig 2: angle overlays (A vs A2, B vs B2, first 400 ep) ───────────────
    SW_OV = 25
    fig2, axes2 = plt.subplots(N_CONDS, 2, figsize=(14, 4 * N_CONDS))
    for row, (cond, color, curve) in enumerate(zip(CONDITIONS, colors, curves_list)):
        angle_phases = _split(curve["first_angle"])
        for col, (idx1, idx2, target, title) in enumerate([
            (1, 3, -45, "A vs A2  (rot+45°)"),
            (2, 4,   0, "B vs B2  (curl+)"),
        ]):
            ax = axes2[row, col]
            for idx, alpha_val, label in [
                (idx1, 0.45, PHASE_LABELS[idx1]),
                (idx2, 1.00, PHASE_LABELS[idx2]),
            ]:
                y_raw = np.array(angle_phases[idx])
                y_sm  = smooth(y_raw, SW_OV)
                ep_sm = ep_local[SW_OV - 1:] if len(y_raw) >= SW_OV else ep_local[:len(y_sm)]
                mask  = ep_sm <= 400
                ax.plot(ep_sm[mask], y_sm[mask], color=color, lw=2, alpha=alpha_val, label=label)
            ax.axhline(target, color="gray", ls="--", lw=1.2, label=f"{target:+d}° target")
            ax.set_xlim(1, 400); ax.set_ylim(-100, 60)
            ax.legend(fontsize=7, loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0: ax.set_title(title, fontsize=10)
            if col == 0: ax.set_ylabel(f"{cond['name']}\nFirst-step angle (deg)", fontsize=8)
            if row == N_CONDS - 1: ax.set_xlabel("Episode within phase", fontsize=8)
    fig2.suptitle(
        "ABA savings — first-step angle overlays across 4 conditions\n"
        "(faded = first exposure, solid = second exposure)",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_4cond_angle_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_4cond_angle_overlay.png")

    # ── Fig 3: success asymptote fits (A / B / A2) ────────────────────────────
    ABA_INDICES = [1, 2, 3]
    SW_ACC = 20
    fig3, axes3 = plt.subplots(N_CONDS, 3, figsize=(15, 4 * N_CONDS), sharey=True)
    for row, (cond, color, curve) in enumerate(zip(CONDITIONS, colors, curves_list)):
        succ_all = _split(curve["success"])
        for col, phase_idx in enumerate(ABA_INDICES):
            ax = axes3[row, col]
            y_raw = np.array(succ_all[phase_idx])
            y_sm  = smooth(y_raw, SW_ACC)
            ep_sm = ep_local[SW_ACC - 1:] if len(y_raw) >= SW_ACC else ep_local[:len(y_sm)]
            ax.plot(ep_sm, y_sm, color=color, lw=2, alpha=0.45, zorder=1)
            if len(y_sm) > 10:
                L, y0, k, x95 = fit_asymptote(ep_sm, y_sm)
                if L is not None:
                    ax.plot(ep_sm, L + (y0 - L) * np.exp(-k * ep_sm),
                            color="tomato", ls="--", lw=2, zorder=3,
                            label=f"asymptote={L:.2f}")
                if x95 is not None and x95 < N_PHASE_EP:
                    ax.axvline(x95, color="black", ls=":", lw=1.3, zorder=2)
                    ax.text(x95, 0.05, f"ep {int(x95)}", fontsize=7, ha="center", va="bottom")
            ax.set_xlim(1, N_PHASE_EP); ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0: ax.set_title(PHASE_LABELS[phase_idx], fontsize=10)
            if col == 0: ax.set_ylabel(f"{cond['name']}\nSuccess rate", fontsize=8)
            if row == N_CONDS - 1: ax.set_xlabel("Episode within phase", fontsize=8)
    fig3.suptitle("A / B / A2 success asymptote fits — 4 conditions, with routing", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_4cond_acc_asymptote.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_4cond_acc_asymptote.png")

    # ── Fig 4: forgetting probe + savings bar ─────────────────────────────────
    from matplotlib.patches import Patch
    fig4, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))
    SW_PROBE = 30
    for cond, color, curve in zip(CONDITIONS, colors, curves_list):
        probe_succ = np.array(curve.get("probe_a_succ", []))
        probe_ph   = np.array(curve.get("probe_a_phase", []))
        mask = probe_ph == 2
        if mask.sum() > SW_PROBE:
            y_sm = smooth(probe_succ[mask], SW_PROBE)
            axL.plot(np.arange(1, len(y_sm) + 1), y_sm, color=color, lw=2, label=cond["name"])
    axL.set_xlabel("Episode within B phase", fontsize=10)
    axL.set_ylabel("Context-A success rate (probe)", fontsize=10)
    axL.set_title("Does B overwrite A?\n(probe on A during B training)", fontsize=10)
    axL.set_ylim(0, 1.05)
    axL.legend(fontsize=9)
    axL.spines[["top", "right"]].set_visible(False)

    x95_A, x95_A2 = [], []
    for curve in curves_list:
        succ_phases = _split(curve["success"])
        for p_idx, store in [(1, x95_A), (3, x95_A2)]:
            y_sm  = smooth(np.array(succ_phases[p_idx]), 20)
            ep_sm = ep_local[19:] if len(succ_phases[p_idx]) >= 20 else ep_local[:len(y_sm)]
            _, _, _, x95 = fit_asymptote(ep_sm, y_sm)
            store.append(x95 if (x95 is not None and x95 < N_PHASE_EP) else N_PHASE_EP)

    width = 0.35
    xs = np.arange(N_CONDS)
    for i, (cond, color, a, a2) in enumerate(zip(CONDITIONS, colors, x95_A, x95_A2)):
        axR.bar(i - width/2, a,  width, color=color, alpha=0.4)
        axR.bar(i + width/2, a2, width, color=color, alpha=1.0)
        saving = a - a2
        mid_y  = (a + a2) / 2
        axR.annotate("", xy=(i + width/2, a2), xytext=(i + width/2, a),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        axR.text(i + width/2 + 0.07, mid_y, f"Δ={saving:.0f}", fontsize=8, va="center")
    axR.set_xticks(xs)
    axR.set_xticklabels([c["name"] for c in CONDITIONS], fontsize=8, rotation=15, ha="right")
    axR.set_ylabel("Episodes to 95% asymptote", fontsize=10)
    axR.set_title("Savings: A vs A2\n(faded=1st, solid=2nd)", fontsize=10)
    axR.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="A (1st)"),
                         Patch(facecolor="gray", alpha=1.0, label="A2 (2nd)")], fontsize=9)
    axR.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_4cond_forgetting_and_savings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_4cond_forgetting_and_savings.png")

    # ── Fig 5: per-condition savings overlay (separate figures) ───────────────
    SW_OV5  = 20
    EP_SHOW = 300
    for cond, color, curve in zip(CONDITIONS, colors, curves_list):
        angle_phases = _split(curve["first_angle"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (idx1, idx2), target, title in zip(
            axes,
            [(1, 3), (2, 4)],
            [-45,      0],
            ["A vs A2  (rot+45°)", "B vs B2  (curl+)"],
        ):
            for idx, alpha_val, lbl in [
                (idx1, 0.35, PHASE_LABELS[idx1] + " (1st)"),
                (idx2, 1.00, PHASE_LABELS[idx2] + " (2nd)"),
            ]:
                y_raw = np.array(angle_phases[idx])
                y_sm  = smooth(y_raw, SW_OV5)
                ep_sm = ep_local[SW_OV5 - 1:] if len(y_raw) >= SW_OV5 else ep_local[:len(y_sm)]
                mask  = ep_sm <= EP_SHOW
                ax.plot(ep_sm[mask], y_sm[mask], color=color, lw=2.5, alpha=alpha_val, label=lbl)
            ax.axhline(target, color="gray", ls="--", lw=1.2, label=f"{target:+d}° (target)")
            ax.set_xlim(1, EP_SHOW); ax.set_ylim(-100, 60)
            ax.set_xlabel("Episode within phase", fontsize=10)
            ax.set_ylabel("First-step angle (deg)", fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=8, loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Savings overlay — {cond['name']}\n"
                     f"(faded=1st, solid=2nd | first {EP_SHOW} eps | rolling mean w={SW_OV5})",
                     fontsize=11)
        plt.tight_layout()
        fname = f"savings_overlay_{cond['name'].replace(' ', '_').replace('+', 'p')}.png"
        plt.savefig(f"{FIG_DIR}/{fname}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen EC + CB
    ckpt = torch.load(CKPT_ECTOCB, map_location=device)
    ec   = EntorhinalCortex().to(device)
    cb   = CerebellumCB().to(device)
    ec.load_state_dict(ckpt["ec"])
    remap  = {"net.0": "layer1.0", "net.2": "layer2.0", "net.4": "out"}
    cb_sd  = {k.replace(".".join(k.split(".")[:2]),
               remap[".".join(k.split(".")[:2])], 1)
              if ".".join(k.split(".")[:2]) in remap else k: v
              for k, v in ckpt["cb"].items()}
    cb.load_state_dict(cb_sd)
    ec.eval(); cb.eval()
    print(f"Loaded EC + CB from {CKPT_ECTOCB}")

    # Load pretrained thalamus weights (used for routing conditions)
    thal_ckpt = torch.load(CKPT_THAL, map_location=device)

    os.makedirs(FIG_DIR, exist_ok=True)
    data_path = f"{FIG_DIR}/routing_savings_data.npz"

    curves_list = []
    cond_names  = [c["name"] for c in CONDITIONS]

    for cond in CONDITIONS:
        name        = cond["name"]
        use_ec      = cond["use_ec"]
        use_cb      = cond["use_cb"]
        use_routing = cond["use_routing"]
        retrain_cb  = cond["retrain_cb"]

        print(f"\n{'='*60}")
        print(f"Condition: {name}")
        print(f"{'='*60}")

        mc      = MotorCortexMoE(use_cb=use_cb, use_routing=use_routing).to(device)
        cb_noec = CerebellumCB_NoEC().to(device) if retrain_cb else None

        if use_routing:
            mc.thalamus.load_state_dict(thal_ckpt["thalamus_state_dict"])
            for p in mc.thalamus.parameters():
                p.requires_grad_(False)

        curve = run_aba(ec, cb, mc, device, seed=42, use_ec=use_ec, cb_noec=cb_noec)
        curves_list.append(curve)

    np.savez(data_path, curves=np.array(curves_list, dtype=object),
             cond_names=cond_names)
    print(f"\nSaved data to {data_path}")

    plot_all(curves_list, cond_names, COND_COLORS)

    # Quick savings summary
    print("\n── Savings summary (first-50-episode mean angle) ──")
    for curve, name in zip(curves_list, cond_names):
        phase = np.array(curve["phase"])
        fa    = np.array(curve["first_angle"])
        a1    = fa[phase == 1]
        a2    = fa[phase == 3]
        print(f"  {name:20s}  A first50={np.mean(a1[:50]):.1f}°  "
              f"A2 first50={np.mean(a2[:50]):.1f}°  "
              f"diff={np.mean(a2[:50])-np.mean(a1[:50]):.1f}°")

    print("\nDone.")

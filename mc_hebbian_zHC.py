"""
mc_hebbian_zHC.py

HC savings with z_t (EC readout) as key instead of h_ec.

Key change vs MC_HC_hebbian.py
--------------------------------
  OLD: key = cat(h_ec, g) — 258-dim (EC GRU hidden state, state-dominated)
  NEW: key = cat(z_t,   g) —  34-dim (EC tanh readout, more compact)

z_t: (1, LATENT_DIM=32) — tanh-compressed EC readout at each timestep.
     More compact than h_ec, still carries context info via prediction errors.
     KEY_DIM = LATENT_DIM + 2 = 34.

HC architecture otherwise unchanged — A and B still purely Hebbian.
No t>=5 gating — full episode used for HC loss and Hebbian update.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM   = 32
EC_HIDDEN    = 256
CB_HIDDEN    = 256
HC_HIDDEN    = 256
EC_INPUT     = 8
CB_PROJ_DIM  = 32
MC_HIDDEN    = 256

MAX_STEPS    = 20
N_PHASE_EP   = 2000
GOAL_RADIUS  = 0.1
GRID_SIZE    = 4.0
DT           = 1.0
LR_MC        = 1e-3
LR_HC        = 1e-4
HC_KL_WEIGHT = 0.3

CKPT    = "checkpoints/ectocb_8ctx_seq80_z32.pt"
FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/MC_EC_CB/MC_savings_norouting/zHC_hebbian"

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
    {"name": "MC only",               "use_ec": False, "use_cb": False, "retrain_cb_noec": False, "use_hc": False, "ec_lesion": False},
    {"name": "MC + CB",               "use_ec": False, "use_cb": True,  "retrain_cb_noec": True,  "use_hc": False, "ec_lesion": False},
    {"name": "MC + EC",               "use_ec": True,  "use_cb": False, "retrain_cb_noec": False, "use_hc": False, "ec_lesion": False},
    {"name": "MC + EC + CB",          "use_ec": True,  "use_cb": True,  "retrain_cb_noec": False, "use_hc": False, "ec_lesion": False},
    {"name": "MC + EC + CB + HC",     "use_ec": True,  "use_cb": True,  "retrain_cb_noec": False, "use_hc": True,  "ec_lesion": False},
    {"name": "EC lesion (HC active)", "use_ec": True,  "use_cb": True,  "retrain_cb_noec": False, "use_hc": True,  "ec_lesion": True},
    {"name": "EC+HC lesion",          "use_ec": True,  "use_cb": True,  "retrain_cb_noec": False, "use_hc": False, "ec_lesion": True},
]
COND_COLORS = ["dimgray", "steelblue", "seagreen", "tomato", "darkorange", "salmon", "lightcoral"]
KEEP_IDX    = [0, 1, 2, 3, 4, 6]

KEY_DIM = LATENT_DIM + 2  # cat(z_t, g)


# ── Hippocampus ───────────────────────────────────────────────────────────────
HC_ALPHA = 0.1   # mixing coefficient: h_hc = (1-α)·h_hc + α·tanh(A·Z)

class HippocampusRNN(nn.Module):
    """
    Two-step Hopfield RNN: recurrent h_hc integrates across timesteps.
    h_hc(t+1) = (1-α)·h_hc(t) + α·tanh(A·Z_t)   — Hopfield attractor dynamics
    a_hc      = tanh(B·h_hc)                       — action readout
    A: (HC_HIDDEN, KEY_DIM) = (256, 258)  Hebbian: ΔA = h_hc ⊗ Z^T
    B: (2, HC_HIDDEN)       = (2,   256)  Hebbian: ΔB = a*   ⊗ h_hc^T
    h_hc resets to zero each episode.
    """
    def __init__(self):
        super().__init__()
        self.A = nn.Linear(KEY_DIM, HC_HIDDEN, bias=False)  # Z → h
        self.B = nn.Linear(HC_HIDDEN, 2,       bias=False)  # h → action

        nn.init.zeros_(self.A.weight)
        nn.init.zeros_(self.B.weight)

        for p in self.parameters():
            p.requires_grad_(False)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, HC_HIDDEN, device=device)

    def step(self, z_t, g, h_hc):
        key   = torch.cat([z_t, g], dim=-1)                         # (1, KEY_DIM=34)
        h_hc  = (1 - HC_ALPHA) * h_hc + HC_ALPHA * torch.tanh(self.A(key))  # (1, HC_HIDDEN=256)
        a_hc  = torch.tanh(self.B(h_hc))                            # (1, 2)
        return a_hc, h_hc, key


# ── Hebbian update ────────────────────────────────────────────────────────────
def hc_hebbian_update(hc, hc_buffer, lr):
    """
    Two-factor Hebb on A and B after rewarded episode.
    hc_buffer : list of (key_t, h_t, a_mc_t)
      key_t   : (1, KEY_DIM=258)
      h_t     : (1, HC_HIDDEN=256)
      a_mc_t  : (2,)
    ΔA = lr * h ⊗ key^T    — store "key Z evoked HC state h"
    ΔB = lr * a* ⊗ h^T     — store "HC state h produced action a*"
    """
    with torch.no_grad():
        for key_t, h_t, a_mc_t in hc_buffer:
            hc.A.weight.data += lr * torch.outer(h_t.squeeze(0), key_t.squeeze(0))
            hc.B.weight.data += lr * torch.outer(a_mc_t, h_t.squeeze(0))


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
        input_dim   = 2 + 2
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
        h_new = self.cell(torch.cat(parts, dim=-1), h_mc)
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
    hc=None, use_hc=False, ec_lesion=False,
    h_ec_in=None, hc_weight=HC_KL_WEIGHT,
):
    beta, theta, gamma, g, bx, by, _ = ctx

    start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(start - goal) < 0.5:
        goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    g_t  = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos  = torch.tensor(start, dtype=torch.float32, device=device)
    vel  = torch.zeros(2, dtype=torch.float32, device=device)

    h_ec   = h_ec_in if h_ec_in is not None else ec.init_hidden(1, device)
    h_mc   = mc.init_hidden(1, device)
    h_hc   = hc.init_hidden(1, device) if (use_hc and hc is not None) else None
    z_t    = torch.zeros(1, LATENT_DIM, device=device)
    prev_a = torch.zeros(1, 2, device=device)

    goal_losses = []
    hc_losses   = []
    hc_buffer   = []   # (key_t, h_t, a_mc_t) for Hebbian update
    dist_seq    = []
    angle_seq   = []
    success     = False
    steps       = 0
    first_angle = None
    init_dist   = torch.norm(pos - g_t).item()

    for t in range(MAX_STEPS):
        d = torch.norm(pos - g_t).item()
        dist_seq.append(d)
        if d < GOAL_RADIUS:
            success = True
            break

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
            goal_losses.append(torch.sum((g_t - s_next) ** 2))
            if cb_noec is not None:
                pred = cb_noec(pos.unsqueeze(0), a_t.detach())
                goal_losses.append(torch.sum((s_next.detach().unsqueeze(0) - pred) ** 2))

            if use_hc and hc is not None and h_hc is not None:
                with torch.no_grad():
                    a_hc, h_hc, key_t = hc.step(z_t, g_t.unsqueeze(0), h_hc)
                # skip first 4 steps — z_t not yet context-informative
                if t >= 5:
                    # Constraint in MC loss — a_hc detached, no gradient into HC
                    hc_losses.append(
                        hc_weight * torch.sum((a_t.squeeze(0) - a_hc.squeeze(0).detach()) ** 2)
                    )
                    # Store for Hebbian update
                    hc_buffer.append((key_t.detach().clone(), h_hc.detach().clone(), a_t.squeeze(0).detach().clone()))

        with torch.no_grad():
            a_np  = a_t.squeeze(0).detach().cpu().numpy()
            gd    = (g_t - pos).cpu().numpy()
            cross = gd[0] * a_np[1] - gd[1] * a_np[0]
            dot   = a_np[0] * gd[0] + a_np[1] * gd[1]
            alpha = float(np.degrees(np.arctan2(cross, dot)))
            angle_seq.append(alpha)
            if t == 0:
                first_angle = alpha

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

    dist_norm   = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
    mean_angle  = float(np.mean(angle_seq)) if angle_seq else 0.0
    first_angle = first_angle if first_angle is not None else 0.0

    if goal_losses:
        loss = torch.stack(goal_losses).sum()
        if hc_losses:
            loss = loss + torch.stack(hc_losses).sum()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    h_ec_end = h_ec.detach()
    return loss, float(success), steps, dist_norm, mean_angle, first_angle, h_ec_end, hc_buffer


# ── Training loop ─────────────────────────────────────────────────────────────
def run_babab(ec, cb_frozen, mc, device, seed=0,
              use_ec=True, use_cb=False, cb_noec=None,
              use_hc=False, ec_lesion=False):
    ec.eval()
    cb_frozen.eval()
    mc.train()
    for p in ec.parameters():        p.requires_grad_(False)
    for p in cb_frozen.parameters(): p.requires_grad_(False)

    hc = HippocampusRNN().to(device) if use_hc else None

    # HC not in optimizer — updates only via Hebbian rule
    param_groups = [{"params": mc.parameters(), "lr": LR_MC}]
    if cb_noec is not None:
        cb_noec.train()
        param_groups.append({"params": cb_noec.parameters(), "lr": LR_MC})

    optimizer = torch.optim.Adam(param_groups)
    rng       = np.random.default_rng(seed)

    curve = {
        "ep": [], "success": [], "steps": [], "dist_norm": [],
        "angle": [], "first_angle": [], "phase": [],
        "probe_b_ep": [], "probe_b_succ": [],
    }

    ep_global = 0

    for phase_idx, ctx in enumerate(PHASES):
        label = PHASE_LABELS[phase_idx]
        print(f"\n  Phase {label} ({N_PHASE_EP} episodes)...")
        # h_ec resets every episode (h_ec_in=None → ec.init_hidden).
        # EC lesion zeroes h_ec_mc inside run_episode → keys degenerate → no savings.

        for ep in range(1, N_PHASE_EP + 1):
            ep_global += 1

            mc.train()
            optimizer.zero_grad()

            loss, succ, steps, dist_norm, mean_angle, first_angle, _, hc_buffer = run_episode(
                ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
                ctx=ctx, rng=rng, train=True,
                use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec,
                hc=hc, use_hc=use_hc, ec_lesion=ec_lesion,
                h_ec_in=None, hc_weight=HC_KL_WEIGHT,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            optimizer.step()

            # Hebbian update — only on rewarded episodes, no backprop
            if use_hc and hc is not None and succ and hc_buffer:
                hc_hebbian_update(hc, hc_buffer, LR_HC)

            mc.eval()
            if cb_noec is not None:
                cb_noec.eval()

            with torch.no_grad():
                _, succ_e, steps_e, dist_e, ang_e, fang_e, _, _ = run_episode(
                    ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
                    ctx=ctx, rng=rng, train=False,
                    use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec,
                    hc=hc, use_hc=use_hc, ec_lesion=ec_lesion,
                    h_ec_in=None,
                )

            curve["ep"].append(ep_global)
            curve["success"].append(float(succ_e))
            curve["steps"].append(float(steps_e))
            curve["dist_norm"].append(float(dist_e))
            curve["angle"].append(float(ang_e))
            curve["first_angle"].append(float(fang_e))
            curve["phase"].append(phase_idx)

            if phase_idx == B_IDX:
                with torch.no_grad():
                    _, probe_s, _, _, _, _, _, _ = run_episode(
                        ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
                        ctx=CTX_A, rng=rng, train=False,
                        use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec,
                        hc=hc, use_hc=use_hc, ec_lesion=ec_lesion,
                        h_ec_in=None,
                    )
                curve["probe_b_ep"].append(ep_global)
                curve["probe_b_succ"].append(float(probe_s))

            if ep % 500 == 0:
                print(f"    ep {ep:4d} | acc(last100)={np.mean(curve['success'][-100:]):.2f}"
                      f" | angle(last100)={np.mean(curve['angle'][-100:]):.1f}°")

    return curve


# ── Plot helpers ──────────────────────────────────────────────────────────────
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
        succ = np.array(curve["success"])
        y_sm = smooth(succ); ep_g = np.arange(1, len(succ) + 1)
        ax.plot(ep_g[SW - 1:], y_sm, color=color, lw=1.5)
        for xb in phase_boundaries:
            ax.axvline(xb, color="gray", ls="--", lw=0.8)
        for i, lbl in enumerate(PHASE_LABELS):
            ax.text((i + 0.5) * N_PHASE_EP, 1.02, lbl, ha="center", fontsize=7,
                    transform=ax.get_xaxis_transform())
        ax.set_ylim(0, 1.1); ax.set_ylabel(name, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if row == n_conds - 1:
            ax.set_xlabel("Episode (global)", fontsize=9)
    fig.suptitle(f"Hopfield HC — Learning Curves (seed {seed})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/hebbian_learning_curves.png", dpi=150, bbox_inches="tight")
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
            y_sm = smooth(np.array(phases[ph_idx]))
            ep_sm = ep_local[SW - 1:]; mask = ep_sm <= EP_SHOW
            ax.plot(ep_sm[mask], y_sm[mask], color=col, lw=2, label=lbl)
        ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° target")
        ax.set_xlim(1, EP_SHOW); ax.set_ylim(-110, 60)
        ax.legend(fontsize=7, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.set_title("A vs A2 — mean trajectory angle", fontsize=10)
        ax.set_ylabel(f"{name}\nAngle (°)", fontsize=8)
        if row == n_conds - 1:
            ax.set_xlabel("Episode within phase", fontsize=8)
    fig.suptitle(f"Hopfield HC: A vs A2 angle overlay (seed {seed})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/hebbian_angle_overlay.png", dpi=150, bbox_inches="tight")
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
    ax.set_title(f"Hopfield HC Savings: A vs A2 (seed {seed})", fontsize=10)
    ax.set_ylim(0, 1.5)
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="A (1st)"),
                       Patch(facecolor="gray", alpha=1.0, label="A2 (2nd)")], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/hebbian_savings_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 4: EC lesion comparison
    if intact_name in name_to_curve and lesion_name in name_to_curve:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, ph_idx, ph_lbl in [(axes[0], A_IDX, PHASE_LABELS[A_IDX]),
                                    (axes[1], A2_IDX, PHASE_LABELS[A2_IDX])]:
            for name, ls, lw, alpha in [(intact_name, "-", 2.0, 1.0), (lesion_name, "--", 2.0, 0.8)]:
                phases = split_phases(name_to_curve[name]["first_angle"])
                y_sm = smooth(np.array(phases[ph_idx]))
                ep_sm = ep_local[SW - 1:]; mask = ep_sm <= 400
                ax.plot(ep_sm[mask], y_sm[mask], color=name_to_color[name],
                        lw=lw, alpha=alpha, ls=ls, label=name)
            ax.axhline(-60, color="gray", ls=":", lw=1.2)
            ax.set_xlim(1, 400); ax.set_ylim(-110, 60)
            ax.set_title(ph_lbl, fontsize=10)
            ax.set_xlabel("Episode within phase", fontsize=9)
            ax.set_ylabel("Mean trajectory angle (°)", fontsize=9)
            ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"EC+HC lesion vs intact (seed {seed})", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/hebbian_ec_lesion.png", dpi=150, bbox_inches="tight")
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
            y_sm = smooth(np.array(phases[ph_idx]))
            ep_sm = ep_local[SW - 1:]; mask = ep_sm <= EP_SHOW
            ax.plot(ep_sm[mask], y_sm[mask], color=col, ls=ls, lw=2, label=lbl)
        ax.axhline(-60, color="gray", ls=":", lw=1.2)
        ax.set_xlim(1, EP_SHOW); ax.set_ylim(-110, 60)
        ax.set_xlabel("Episode within phase", fontsize=9)
        ax.set_ylabel("Mean trajectory angle (°)", fontsize=9)
        ax.set_title(f"Intact vs EC+HC lesion — A & A2 (seed {seed})", fontsize=10)
        ax.legend(fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/hebbian_ec_lesion_overlay.png", dpi=150, bbox_inches="tight")
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
        plt.savefig(f"{fig_dir}/hebbian_aba_accuracy_lesion.png", dpi=150, bbox_inches="tight")
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
        plt.savefig(f"{fig_dir}/hebbian_savings_accuracy.png", dpi=150, bbox_inches="tight")
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
        plt.savefig(f"{fig_dir}/hebbian_savings_raw_fit.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Fig 6d: Angle slope
    if intact_name in name_to_curve and lesion_name in name_to_curve:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, name, title in [(axes[0], intact_name, "Intact"), (axes[1], lesion_name, "EC+HC lesion")]:
            angle_phases = split_phases(name_to_curve[name]["first_angle"])
            for ph_idx, col, lbl in [(A_IDX, "steelblue", "A (1st)"), (A2_IDX, "darkorange", "A2 (2nd)")]:
                y_sm = smooth(np.array(angle_phases[ph_idx]), 20)
                ep_sm = ep_local[20 - 1:]
                dy = np.diff(y_sm); dy_sm = smooth(dy, 20)
                ep_dy = ep_sm[:len(dy_sm)]; mask = ep_dy <= 150
                ax.plot(ep_dy[mask], dy_sm[mask], color=col, lw=1.5, label=lbl)
            ax.axhline(0, color="gray", ls="--", lw=0.8); ax.set_xlim(1, 150)
            ax.set_xlabel("Episode within phase", fontsize=9)
            ax.set_ylabel("d(angle)/d(episode)", fontsize=9)
            ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Slope of angle descent (seed {seed})", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/hebbian_angle_slope.png", dpi=150, bbox_inches="tight")
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
        plt.savefig(f"{fig_dir}/hebbian_savings_slope.png", dpi=150, bbox_inches="tight")
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
                ax.plot(ep_sm_fit, exp_model(ep_sm_fit, L, y0, k),
                        color="black", lw=1.2, ls="--", alpha=0.7)
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
    plt.savefig(f"{fig_dir}/hebbian_aba_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Seed {seed}: all 11 figures saved to {fig_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────
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
    ec.eval(); cb_frozen.eval()
    print(f"Loaded EC + CB from {CKPT}")

    for seed in range(8):
        seed_dir = os.path.join(FIG_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        curves_all = []
        for cond_idx, cond in enumerate(CONDITIONS):
            cond_path = os.path.join(seed_dir, f"cond_{cond_idx}.npz")
            if os.path.exists(cond_path):
                print(f"Condition {cond['name']}: data exists, skipping.")
                d = np.load(cond_path, allow_pickle=True)
                curves_all.append(d["curve"].item())
            else:
                print(f"\n{'=' * 60}")
                print(f"Seed {seed} | Condition: {cond['name']}")
                print(f"{'=' * 60}")
                mc      = MotorCortex(use_ec=cond["use_ec"], use_cb=cond["use_cb"]).to(device)
                cb_noec = CerebellumCB_NoEC().to(device) if cond["retrain_cb_noec"] else None
                curve = run_babab(
                    ec=ec, cb_frozen=cb_frozen, mc=mc, device=device, seed=seed,
                    use_ec=cond["use_ec"], use_cb=cond["use_cb"], cb_noec=cb_noec,
                    use_hc=cond["use_hc"], ec_lesion=cond["ec_lesion"],
                )
                curves_all.append(curve)
                np.savez(cond_path, curve=curve)
                print(f"  Saved condition {cond_idx} to {cond_path}")

        np.savez(os.path.join(seed_dir, "hebbian_data.npz"), curves=curves_all)
        plot_seed(seed, curves_all, seed_dir)
        print(f"Seed {seed} done.")


if __name__ == "__main__":
    main()

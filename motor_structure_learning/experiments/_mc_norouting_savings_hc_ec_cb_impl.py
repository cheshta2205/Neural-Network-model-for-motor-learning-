"""
MC_norouting_savings_HC_EC_CB.py

Same as MC_norouting_savings (ABA, 4 conditions, no thalamus routing)
but whenever EC is present, a Hippocampus (HC) module is added.

EC no longer feeds z_ec directly into MC.
Instead: EC → h_ec → HC (non-parametric KNN memory) → h_mc_retrieved → MC init state.

HC:
  - Stores { key: h_ec_final (256), value: h_mc_final (256) } at episode end
  - Reads  { query: h_ec_current (256) } → weighted average of stored h_mc → h_mc_retrieved
  - h_mc_retrieved initialises MC's hidden state at episode start (instead of zeros)
  - Non-parametric: no weights, no gradient, no loss

Conditions:
  1. MC only          — no EC, no CB, no HC
  2. MC + CB          — no EC, CB hidden to MC, retrain CB_NoEC, no HC
  3. MC + EC + HC     — EC → HC → MC (h_mc init), no CB to MC
  4. MC + CB + EC + HC — EC → HC → MC (h_mc init) + CB hidden to MC

Savings mechanism:
  - CB: better error signal in A2 (faster re-learning)
  - HC+EC: better MC initial state in A2 (immediate recall)
  - Together: additive savings

Figures saved to: figs/ECtoCB/MC_EC_CB/MC_savings_HC_60curl60_ABA/
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM  = 32
EC_HIDDEN   = 256
CB_HIDDEN   = 256
EC_INPUT    = 8
CB_PROJ_DIM = 32
MC_HIDDEN   = 256

HC_CAPACITY    = 5000   # max episodes stored in HC
HC_TEMPERATURE = 0.05   # softmax temperature for retrieval (lower = sharper)

MAX_STEPS   = 20
N_PHASE_EP  = 2000

GOAL_RADIUS = 0.1
GRID_SIZE   = 4.0
DT          = 1.0
LR_MC       = 1e-3

CKPT    = "checkpoints/ectocb_8ctx_seq80_z32.pt"
FIG_DIR = "../../figs/ECtoCB/MC_EC_CB/MC_savings_HC_60curl60_ABA"

# ── Contexts / phases ─────────────────────────────────────────────────────────
CTX_A  = (1.0, np.pi/3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B  = (1.0, 0.0,    +0.4, 0.0, 0.0, 0.0, "B: curl+")

PHASES       = [CTX_A, CTX_B, CTX_A]
PHASE_LABELS = ["A: rot+60°", "B: curl+", "A2: rot+60°"]

# ── Conditions ────────────────────────────────────────────────────────────────
CONDITIONS = [
    {"name": "MC only",           "use_ec": False, "use_cb": False, "retrain_cb_noec": False},
    {"name": "MC + CB",           "use_ec": False, "use_cb": True,  "retrain_cb_noec": True},
    {"name": "MC + EC + HC",      "use_ec": True,  "use_cb": False, "retrain_cb_noec": False},
    {"name": "MC + CB + EC + HC", "use_ec": True,  "use_cb": True,  "retrain_cb_noec": False},
]

COND_COLORS = ["dimgray", "steelblue", "seagreen", "tomato"]


# ── Models ────────────────────────────────────────────────────────────────────
class EntorhinalCortex(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = EC_HIDDEN
        self.cell = nn.GRUCell(EC_INPUT, EC_HIDDEN)
        self.readout = nn.Sequential(
            nn.Linear(EC_HIDDEN, LATENT_DIM),
            nn.Tanh()
        )

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, EC_HIDDEN, device=device)

    def step(self, x, h):
        h_new = self.cell(x, h)
        return self.readout(h_new), h_new


class CerebellumCB(nn.Module):
    """Original frozen CB: takes [s, a, z_ec]."""
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
    """CB retrained without EC input: takes [s, a]. Used only in MC+CB."""
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
    """
    MC with no direct EC input.
    When use_ec=True, HC initialises MC's hidden state — EC talks to MC only via HC.
    When use_cb=True, CB hidden (projected) is concatenated to MC input.
    """
    def __init__(self, use_ec=True, use_cb=True):
        super().__init__()
        self.use_ec = use_ec  # bookkeeping only — does not change MC input
        self.use_cb = use_cb

        input_dim = 2 + 2   # state + goal
        if use_cb:
            input_dim += CB_PROJ_DIM
            W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
            self.register_buffer("cb_proj", W / W.norm(dim=0, keepdim=True))

        self.hidden_dim = MC_HIDDEN
        self.cell    = nn.GRUCell(input_dim, MC_HIDDEN)
        self.readout = nn.Linear(MC_HIDDEN, 2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, g, h_mc, cb_hidden):
        """No h_ec argument — EC influence comes only through HC-initialised h_mc."""
        parts = [s_t]
        if self.use_cb:
            z_cb = cb_hidden @ self.cb_proj
            parts.append(z_cb)
        parts.append(g)
        inp   = torch.cat(parts, dim=-1)
        h_new = self.cell(inp, h_mc)
        return torch.tanh(self.readout(h_new)), h_new


# ── Hippocampus (non-parametric KNN episodic memory) ─────────────────────────
class Hippocampus:
    """
    Non-parametric key-value episodic memory.

    Key:   h_ec_final  (EC hidden state at episode end, 256-dim)
    Value: h_mc_final  (MC hidden state at episode end, 256-dim)

    Write: called at end of each training episode.
    Read:  called at start of each episode — returns weighted avg of stored h_mc
           (used to initialise MC hidden state).
    """
    def __init__(self, ec_dim=EC_HIDDEN, mc_dim=MC_HIDDEN,
                 capacity=HC_CAPACITY, temperature=HC_TEMPERATURE):
        self.ec_dim      = ec_dim
        self.mc_dim      = mc_dim
        self.capacity    = capacity
        self.temperature = temperature
        self.keys   = []   # list of (1, EC_HIDDEN) cpu tensors
        self.values = []   # list of (1, MC_HIDDEN) cpu tensors

    def write(self, h_ec, h_mc):
        self.keys.append(h_ec.detach().cpu())
        self.values.append(h_mc.detach().cpu())
        if len(self.keys) > self.capacity:
            self.keys.pop(0)
            self.values.pop(0)

    def read(self, h_ec_query, device):
        """Return h_mc_retrieved (1, MC_HIDDEN). Returns zeros if buffer empty."""
        if len(self.keys) == 0:
            return torch.zeros(1, self.mc_dim, device=device)
        keys_t = torch.cat(self.keys,   dim=0).to(device)   # (N, EC_HIDDEN)
        vals_t = torch.cat(self.values, dim=0).to(device)   # (N, MC_HIDDEN)
        query  = F.normalize(h_ec_query, dim=-1)            # (1, EC_HIDDEN)
        keys_n = F.normalize(keys_t,     dim=-1)            # (N, EC_HIDDEN)
        sims   = (keys_n @ query.T).squeeze(-1)             # (N,)
        weights = F.softmax(sims / self.temperature, dim=0) # (N,)
        retrieved = (weights.unsqueeze(-1) * vals_t).sum(0, keepdim=True)  # (1, MC_HIDDEN)
        return retrieved.detach()


# ── Environment step ──────────────────────────────────────────────────────────
def env_step_torch(pos, vel, action, beta, theta, gamma, g, bx, by, device):
    cos_t = torch.tensor(float(np.cos(theta)), dtype=torch.float32, device=device)
    sin_t = torch.tensor(float(np.sin(theta)), dtype=torch.float32, device=device)
    R = torch.stack([
        torch.stack([cos_t, -sin_t]),
        torch.stack([sin_t,  cos_t]),
    ])
    curl  = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32, device=device)
    b_vec = torch.tensor([float(bx), float(by)], dtype=torch.float32, device=device)
    F_    = beta * (R @ action) + gamma * (curl @ vel) + g * b_vec
    vel_new = vel + F_
    pos_new = torch.clamp(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new.detach()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_cb_hidden(cb_frozen, cb_noec, s, a, z, device, use_cb):
    if not use_cb:
        return torch.zeros(1, CB_HIDDEN, device=device)
    if cb_noec is not None:
        return cb_noec.hidden(s, a)
    return cb_frozen.hidden(s, a, z)


def get_cb_pred(cb_frozen, cb_noec, s, a, z):
    if cb_noec is not None:
        return cb_noec(s, a)
    return cb_frozen(s, a, z)


# ── Single episode ────────────────────────────────────────────────────────────
def run_episode(
    ec,
    cb_frozen,
    mc,
    device,
    ctx,
    rng,
    train=True,
    use_ec=True,
    use_cb=True,
    cb_noec=None,
    h_ec_in=None,
    z_t_in=None,
    h_mc_init=None,    # HC-retrieved initial hidden state for MC
):
    beta, theta, gamma, g, bx, by, _ = ctx

    start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(start - goal) < 0.5:
        goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    g_t  = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos  = torch.tensor(start, dtype=torch.float32, device=device)
    vel  = torch.zeros(2, dtype=torch.float32, device=device)

    h_ec   = h_ec_in    if h_ec_in    is not None else ec.init_hidden(1, device)
    z_t    = z_t_in     if z_t_in     is not None else torch.zeros(1, LATENT_DIM, device=device)
    h_mc   = h_mc_init  if h_mc_init  is not None else mc.init_hidden(1, device)
    prev_a = torch.zeros(1, 2, device=device)

    step_losses = []
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

        with torch.no_grad() if not train else torch.enable_grad():
            cb_h = get_cb_hidden(
                cb_frozen=cb_frozen, cb_noec=cb_noec,
                s=pos.unsqueeze(0), a=prev_a, z=z_t,
                device=device, use_cb=use_cb,
            )

        # MC step — no h_ec argument, EC influence only via h_mc_init
        a_t, h_mc = mc.step(pos.unsqueeze(0), g_t.unsqueeze(0), h_mc, cb_h)

        s_next, vel_next = env_step_torch(
            pos, vel, a_t.squeeze(0), beta, theta, gamma, g, bx, by, device
        )

        if train:
            step_losses.append(torch.sum((g_t - s_next) ** 2))
            if cb_noec is not None:
                pred = cb_noec(pos.unsqueeze(0), a_t.detach())
                step_losses.append(torch.sum((s_next.detach().unsqueeze(0) - pred) ** 2))

        with torch.no_grad():
            a_np  = a_t.squeeze(0).detach().cpu().numpy()
            gd    = (g_t - pos).cpu().numpy()
            cross = gd[0] * a_np[1] - gd[1] * a_np[0]
            dot   = a_np[0] * gd[0] + a_np[1] * gd[1]
            alpha = float(np.degrees(np.arctan2(cross, dot)))
            angle_seq.append(alpha)
            if t == 0:
                first_angle = alpha

        # EC update
        with torch.no_grad():
            if use_ec:
                s_tm1    = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                cb_pred  = get_cb_pred(cb_frozen=cb_frozen, cb_noec=cb_noec,
                                       s=pos.unsqueeze(0), a=a_t.detach(), z=z_t)
                eps      = s_next.detach().unsqueeze(0) - cb_pred
                z_t, h_ec = ec.step(
                    torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec
                )

        prev_a = a_t.detach()
        pos    = s_next.detach()
        vel    = vel_next
        steps += 1

    dist_norm   = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
    mean_angle  = float(np.mean(angle_seq)) if len(angle_seq) > 0 else 0.0
    first_angle = first_angle if first_angle is not None else 0.0

    loss = (
        torch.stack(step_losses).sum()
        if len(step_losses) > 0
        else torch.tensor(0.0, device=device, requires_grad=True)
    )

    return (loss, float(success), steps, dist_norm, mean_angle, first_angle,
            h_ec.detach(), z_t.detach(), h_mc.detach())


# ── Training loop ─────────────────────────────────────────────────────────────
def run_aba(ec, cb_frozen, mc, device, seed=0, use_ec=True, use_cb=True, cb_noec=None):
    ec.eval()
    cb_frozen.eval()
    mc.train()

    for p in ec.parameters():
        p.requires_grad_(False)
    for p in cb_frozen.parameters():
        p.requires_grad_(False)

    params = list(mc.parameters())
    if cb_noec is not None:
        cb_noec.train()
        params += list(cb_noec.parameters())

    optimizer = torch.optim.Adam(params, lr=LR_MC)
    rng = np.random.default_rng(seed)

    # HC: only created when EC is present
    hc = Hippocampus() if use_ec else None

    curve = {
        "ep": [], "success": [], "steps": [], "dist_norm": [],
        "angle": [], "first_angle": [], "phase": [],
        "probe_a_ep": [], "probe_a_succ": [], "probe_a_phase": [],
        "mc_state": None, "cb_noec_state": None,
    }

    ep_global  = 0
    h_ec_carry = ec.init_hidden(1, device)
    z_carry    = torch.zeros(1, LATENT_DIM, device=device)

    for phase_idx, ctx in enumerate(PHASES):
        label = PHASE_LABELS[phase_idx]
        print(f"\n  Phase {label} ({N_PHASE_EP} episodes)...")

        for ep in range(1, N_PHASE_EP + 1):
            ep_global += 1

            mc.train()
            if cb_noec is not None:
                cb_noec.train()

            # ── HC read: initialise MC hidden state from retrieved memory ──
            h_mc_init = None
            if hc is not None:
                h_mc_init = hc.read(h_ec_carry, device)

            optimizer.zero_grad()
            (loss, _, _, _, _, _,
             h_ec_carry, z_carry, h_mc_final) = run_episode(
                ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
                ctx=ctx, rng=rng, train=True,
                use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec,
                h_ec_in=h_ec_carry if use_ec else None,
                z_t_in=z_carry     if use_ec else None,
                h_mc_init=h_mc_init,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            # ── HC write: store h_ec and h_mc at episode end ──
            if hc is not None:
                hc.write(h_ec_carry, h_mc_final)

            mc.eval()
            if cb_noec is not None:
                cb_noec.eval()

            with torch.no_grad():
                h_mc_init_eval = hc.read(h_ec_carry, device) if hc is not None else None
                (_, succ, steps, dist_norm, mean_angle, first_angle,
                 _, _, _) = run_episode(
                    ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
                    ctx=ctx, rng=rng, train=False,
                    use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec,
                    h_ec_in=h_ec_carry if use_ec else None,
                    z_t_in=z_carry     if use_ec else None,
                    h_mc_init=h_mc_init_eval,
                )

            curve["ep"].append(ep_global)
            curve["success"].append(float(succ))
            curve["steps"].append(float(steps))
            curve["dist_norm"].append(float(dist_norm))
            curve["angle"].append(float(mean_angle))
            curve["first_angle"].append(float(first_angle))
            curve["phase"].append(phase_idx)

            if phase_idx in [1]:
                with torch.no_grad():
                    h_mc_probe = hc.read(h_ec_carry, device) if hc is not None else None
                    (_, probe_succ, _, _, _, _, _, _, _) = run_episode(
                        ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
                        ctx=CTX_A, rng=rng, train=False,
                        use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec,
                        h_ec_in=h_ec_carry if use_ec else None,
                        z_t_in=z_carry     if use_ec else None,
                        h_mc_init=h_mc_probe,
                    )
                curve["probe_a_ep"].append(ep_global)
                curve["probe_a_succ"].append(float(probe_succ))
                curve["probe_a_phase"].append(phase_idx)

            if ep % 500 == 0:
                recent_acc = np.mean(curve["success"][-100:])
                recent_ang = np.mean(curve["angle"][-100:])
                print(
                    f"    ep {ep:4d} | acc(last100)={recent_acc:.2f} "
                    f"| mean_angle(last100)={recent_ang:.1f}°"
                )

    curve["mc_state"] = {k: v.detach().cpu().clone() for k, v in mc.state_dict().items()}
    if cb_noec is not None:
        curve["cb_noec_state"] = {
            k: v.detach().cpu().clone() for k, v in cb_noec.state_dict().items()
        }

    return curve


# ── Plot helpers ──────────────────────────────────────────────────────────────
def smooth(x, w):
    x = np.array(x)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fit_asymptote(x, y):
    from scipy.optimize import curve_fit

    def model(xv, L, y0, k):
        return L + (y0 - L) * np.exp(-k * xv)

    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        popt, _ = curve_fit(model, x, y, p0=[float(y[-1]), float(y[0]), 1e-3], maxfev=5000)
        L, y0, k = popt
        x_95 = -np.log(0.05) / k if (k > 0 and abs(L - y0) > 1e-6) else None
        return L, y0, k, x_95
    except Exception:
        return None, None, None, None


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_all(curves):
    os.makedirs(FIG_DIR, exist_ok=True)

    N_PHASES = len(PHASES)
    N_CONDS  = len(CONDITIONS)
    ep_local = np.arange(1, N_PHASE_EP + 1)

    def split_phases(arr):
        arr = np.array(arr)
        return [arr[i * N_PHASE_EP:(i + 1) * N_PHASE_EP] for i in range(N_PHASES)]

    phase_boundaries = [i * N_PHASE_EP for i in range(1, N_PHASES)]

    # Fig 1 — learning curves
    fig1, axes1 = plt.subplots(N_CONDS, 3, figsize=(22, 4 * N_CONDS), sharey="col")
    SW_SUCC = SW_STEPS = SW_DIST = 20
    metrics = [
        ("success",   "Success rate",            SW_SUCC,  True),
        ("steps",     "Mean steps to goal",      SW_STEPS, True),
        ("dist_norm", "Mean norm. dist to goal", SW_DIST,  True),
    ]

    for row, (cond, color, curve) in enumerate(zip(CONDITIONS, COND_COLORS, curves)):
        for col, (metric, ylabel, sw, do_smooth) in enumerate(metrics):
            ax = axes1[row, col]
            phases = split_phases(curve[metric])
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
            for i in range(N_PHASES):
                for t in [0, 1000, 2000]:
                    tick_pos.append(i * N_PHASE_EP + t)
                    tick_lbl.append(str(t))
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, fontsize=6)
            if row == 0:
                ax.set_title(ylabel, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{cond['name']}\n{ylabel}", fontsize=8)
            if row == N_CONDS - 1:
                ax.set_xlabel("Episode within phase", fontsize=8)
            if metric == "success":
                ax.set_ylim(0, 1.05)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                y_top = ax.get_ylim()[1]
                for i, lbl in enumerate(PHASE_LABELS):
                    mid = i * N_PHASE_EP + N_PHASE_EP // 2
                    ax.text(mid, y_top, lbl, ha="center", va="bottom", fontsize=6,
                            color="dimgray",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    fig1.suptitle(
        "ABA savings — 4 conditions (EC→HC→MC)\n"
        "(rolling means; black dashed on success = exponential fit)", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_4cond_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_4cond_learning_curves.png")

    # Fig 2 — angle overlay A vs A2
    SW_OV = 25
    fig2, axes2 = plt.subplots(N_CONDS, 1, figsize=(7, 4 * N_CONDS), squeeze=False)
    for row, (cond, color, curve) in enumerate(zip(CONDITIONS, COND_COLORS, curves)):
        angle_phases = split_phases(curve["first_angle"])
        ax = axes2[row, 0]
        for idx, alpha_val, label in [
            (0, 0.45, PHASE_LABELS[0]),
            (2, 1.00, PHASE_LABELS[2]),
        ]:
            y_raw = np.array(angle_phases[idx])
            y_sm  = smooth(y_raw, SW_OV)
            ep_sm = ep_local[SW_OV - 1:] if len(y_raw) >= SW_OV else ep_local[:len(y_sm)]
            ax.plot(ep_sm[ep_sm <= 400], y_sm[ep_sm <= 400],
                    color=color, lw=2, alpha=alpha_val, label=label)
        ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° target")
        ax.set_xlim(1, 400)
        ax.set_ylim(-100, 60)
        ax.legend(fontsize=7, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.set_title("A vs A2  (rot+60°)", fontsize=10)
        ax.set_ylabel(f"{cond['name']}\nFirst-step angle (deg)", fontsize=8)
        if row == N_CONDS - 1:
            ax.set_xlabel("Episode within phase", fontsize=8)

    fig2.suptitle(
        "ABA savings — first-step angle overlays\n(faded=A, solid=A2)", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_4cond_angle_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_4cond_angle_overlay.png")

    # Fig 3 — asymptote fits
    ABA_INDICES = [0, 1, 2]
    ABA_LABELS  = [PHASE_LABELS[i] for i in ABA_INDICES]
    SW_ACC = 20
    fig3, axes3 = plt.subplots(N_CONDS, 3, figsize=(15, 4 * N_CONDS), sharey=True)
    for row, (cond, color, curve) in enumerate(zip(CONDITIONS, COND_COLORS, curves)):
        succ_all = split_phases(curve["success"])
        for col, (phase_idx, lbl) in enumerate(zip(ABA_INDICES, ABA_LABELS)):
            ax = axes3[row, col]
            y_raw = np.array(succ_all[phase_idx])
            y_sm  = smooth(y_raw, SW_ACC)
            ep_sm = ep_local[SW_ACC - 1:] if len(y_raw) >= SW_ACC else ep_local[:len(y_sm)]
            ax.plot(ep_sm, y_sm, color=color, lw=2, alpha=0.45, zorder=1)
            if len(y_sm) > 10:
                L, y0, k, x95 = fit_asymptote(ep_sm, y_sm)
                if L is not None and k is not None:
                    ax.plot(ep_sm, L + (y0 - L) * np.exp(-k * ep_sm),
                            color="tomato", ls="--", lw=2, zorder=3, label=f"asymptote={L:.2f}")
                if x95 is not None and x95 < N_PHASE_EP:
                    ax.axvline(x95, color="black", ls=":", lw=1.3, zorder=2)
                    ax.text(x95, 0.05, f"ep {int(x95)}", fontsize=7, ha="center", va="bottom")
            ax.set_xlim(1, N_PHASE_EP)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(lbl, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{cond['name']}\nSuccess rate", fontsize=8)
            if row == N_CONDS - 1:
                ax.set_xlabel("Episode within phase", fontsize=8)

    fig3.suptitle("A / B / A2 success asymptote fits — 4 conditions (EC→HC→MC)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_4cond_acc_asymptote.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_4cond_acc_asymptote.png")

    # Fig 4 — forgetting probe + savings bar
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    axL, axR = axes4
    SW_PROBE = 30
    for cond, color, curve in zip(CONDITIONS, COND_COLORS, curves):
        probe_succ = np.array(curve.get("probe_a_succ", []))
        probe_ph   = np.array(curve.get("probe_a_phase", []))
        mask = probe_ph == 1
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
    for curve in curves:
        succ_phases = split_phases(curve["success"])
        for phase_idx, store in [(0, x95_A), (2, x95_A2)]:
            y_sm  = smooth(np.array(succ_phases[phase_idx]), 20)
            ep_sm = ep_local[19:] if len(succ_phases[phase_idx]) >= 20 else ep_local[:len(y_sm)]
            _, _, _, x95 = fit_asymptote(ep_sm, y_sm)
            store.append(x95 if (x95 is not None and x95 < N_PHASE_EP) else N_PHASE_EP)

    width = 0.35
    xs = np.arange(len(CONDITIONS))
    for i, (cond, color, a, a2) in enumerate(zip(CONDITIONS, COND_COLORS, x95_A, x95_A2)):
        norm_a2 = a2 / a if a > 0 else 1.0
        axR.bar(i - width / 2, 1.0,    width, color=color, alpha=0.4)
        axR.bar(i + width / 2, norm_a2, width, color=color, alpha=1.0)
        saving = 1.0 - norm_a2
        axR.annotate("", xy=(i + width / 2, norm_a2), xytext=(i + width / 2, 1.0),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        axR.text(i + width / 2 + 0.07, (1.0 + norm_a2) / 2,
                 f"{saving:.2f}", fontsize=8, va="center")
    axR.set_xticks(xs)
    axR.set_xticklabels([c["name"] for c in CONDITIONS], fontsize=8, rotation=15, ha="right")
    axR.set_ylabel("Episodes to 95% (normalised to A)", fontsize=10)
    axR.set_title("Savings: A vs A2\n(faded=A=1.0, solid=A2 normalised)", fontsize=10)
    from matplotlib.patches import Patch
    axR.legend(handles=[
        Patch(facecolor="gray", alpha=0.4, label="A (1st)"),
        Patch(facecolor="gray", alpha=1.0, label="A2 (2nd)"),
    ], fontsize=9)
    axR.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_4cond_forgetting_and_savings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_4cond_forgetting_and_savings.png")

    # Fig 5 — per-condition savings overlay
    SW_OV5 = 20
    EP_SHOW = 300
    for cond, color, curve in zip(CONDITIONS, COND_COLORS, curves):
        angle_phases = split_phases(curve["first_angle"])
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        for idx, alpha_val, lbl in [
            (0, 0.35, PHASE_LABELS[0] + " (1st)"),
            (2, 1.00, PHASE_LABELS[2] + " (2nd)"),
        ]:
            y_raw = np.array(angle_phases[idx])
            y_sm  = smooth(y_raw, SW_OV5)
            ep_sm = ep_local[SW_OV5 - 1:] if len(y_raw) >= SW_OV5 else ep_local[:len(y_sm)]
            mask  = ep_sm <= EP_SHOW
            ax.plot(ep_sm[mask], y_sm[mask], color=color, lw=2.5, alpha=alpha_val, label=lbl)
        ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° (target)")
        ax.set_xlim(1, EP_SHOW)
        ax.set_ylim(-100, 60)
        ax.set_xlabel("Episode within phase", fontsize=10)
        ax.set_ylabel("First-step angle (deg)", fontsize=10)
        ax.set_title("A vs A2  (rot+60°)", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Savings overlay — {cond['name']}\n"
                     f"(faded=1st, solid=2nd | first {EP_SHOW} eps | w={SW_OV5})", fontsize=11)
        plt.tight_layout()
        fname = f"savings_overlay_{cond['name'].replace(' ', '_').replace('+', 'p')}.png"
        plt.savefig(f"{FIG_DIR}/{fname}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")


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

    ec.eval()
    cb_frozen.eval()
    print(f"Loaded frozen EC + original CB from {CKPT}")

    os.makedirs(FIG_DIR, exist_ok=True)
    data_path = f"{FIG_DIR}/savings_4cond_data.npz"

    if os.path.exists(data_path) and "--replot" in sys.argv:
        print(f"Loading saved data from {data_path} — skipping training.")
        d      = np.load(data_path, allow_pickle=True)
        curves = list(d["curves"])
    else:
        curves = []
        for cond in CONDITIONS:
            print(f"\n{'=' * 60}")
            print(f"Condition: {cond['name']}")
            print(f"{'=' * 60}")

            mc      = MotorCortex(use_ec=cond["use_ec"], use_cb=cond["use_cb"]).to(device)
            cb_noec = CerebellumCB_NoEC().to(device) if cond["retrain_cb_noec"] else None

            curve = run_aba(
                ec=ec, cb_frozen=cb_frozen, mc=mc, device=device, seed=0,
                use_ec=cond["use_ec"], use_cb=cond["use_cb"], cb_noec=cb_noec,
            )
            curves.append(curve)

        np.savez(data_path, curves=curves)
        print(f"\nSaved data to {data_path}")

    plot_all(curves)
    print("Done.")


if __name__ == "__main__":
    main()

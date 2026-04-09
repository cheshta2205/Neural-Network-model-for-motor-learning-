"""
MC_savings_all.py

4-condition ABA savings experiment comparing:
  1. MC only          — no EC, no CB input to MC
  2. MC + CB          — CB retrained jointly (no z_ec), no EC
  3. MC + EC          — EC intact, no CB→MC signal
  4. MC + EC + CB     — full system

Figures:
  - 4×3 : learning curves (success, steps, dist) per condition
  - 4×2 : angle savings overlay (A vs A2, B vs B2) per condition
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

MAX_STEPS   = 20
N_PHASE_EP  = 2000
GOAL_RADIUS = 0.1
GRID_SIZE   = 4.0
DT          = 1.0
LR_MC       = 1e-3

CKPT          = "checkpoints/ectocb_8ctx_seq80_z32.pt"
CKPT_THALAMUS = "checkpoints/thalamus_pretrained_ema09.pt"
FIG_DIR       = "../../figs/ECtoCB/MC_policy/MC_savings_all_rot60_rot0"

EMA_ALPHA = 0.9
N_EXPERTS = 2
TAU_START = 1.0
TAU_END   = 0.1

CTX_A  = (1.0, np.pi/3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B  = (1.0, 0.0,     0.0, 0.0, 0.0, 0.0, "B: rot0")
PHASES = [CTX_B, CTX_A, CTX_B, CTX_A, CTX_B]
PHASE_LABELS = ["pre: rot 0°", "A: rot+60°", "B: rot 0°", "A2: rot+60°", "B2: rot 0°"]

# 4 conditions
CONDITIONS = [
    {"name": "MC only",      "use_ec": False, "use_cb": False, "retrain_cb": False, "use_thalamus": False},
    {"name": "MC + CB",      "use_ec": False, "use_cb": True,  "retrain_cb": True,  "use_thalamus": False},
    {"name": "MC + EC",      "use_ec": True,  "use_cb": False, "retrain_cb": False, "use_thalamus": False},
    {"name": "MC + EC + CB", "use_ec": True,  "use_cb": True,  "retrain_cb": False, "use_thalamus": False},
]
COND_COLORS = ["dimgray", "steelblue", "seagreen", "tomato"]


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
    """Original CB — takes [s, a, z_ec] as input. Used frozen in conditions 3, 4."""
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


class CerebellumCB_NoEC(nn.Module):
    """CB retrained without z_ec — takes [s, a] only. Used in condition 2."""
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
    """Flexible MC — input_dim depends on use_ec and use_cb."""
    def __init__(self, use_ec=True, use_cb=True):
        super().__init__()
        self.use_ec = use_ec
        self.use_cb = use_cb
        input_dim   = 2 + 2  # s + g always
        if use_ec: input_dim += LATENT_DIM
        if use_cb: input_dim += CB_PROJ_DIM
        self.hidden_dim = MC_HIDDEN
        if use_ec:
            self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.cell    = nn.GRUCell(input_dim, MC_HIDDEN)
        self.readout = nn.Linear(MC_HIDDEN, 2)
        if use_cb:
            W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
            self.register_buffer("cb_proj", W / W.norm(dim=0, keepdim=True))

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, h_ec, g, h_mc, cb_hidden):
        parts = [s_t]
        if self.use_ec:
            z_ec = torch.tanh(self.ec_readout(h_ec))
            parts.append(z_ec)
        if self.use_cb:
            z_cb = cb_hidden @ self.cb_proj
            parts.append(z_cb)
        parts.append(g)
        inp   = torch.cat(parts, dim=-1)
        h_new = self.cell(inp, h_mc)
        return torch.tanh(self.readout(h_new)), h_new


# ── Thalamus MoE classes ──────────────────────────────────────────────────────
import torch.nn.functional as F

class MCExpert(nn.Module):
    """Single context-specific expert — sees [s_t, g] only; context enters via routing."""
    def __init__(self):
        super().__init__()
        self.hidden_dim = MC_HIDDEN
        self.cell    = nn.GRUCell(2 + 2, MC_HIDDEN)
        self.readout = nn.Linear(MC_HIDDEN, 2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, g, h_mc):
        h_new = self.cell(torch.cat([s_t, g], dim=-1), h_mc)
        return torch.tanh(self.readout(h_new)), h_new


class ThalamusRouter(nn.Module):
    """Routes EC's z_ec to the appropriate MC expert (hard Gumbel-softmax)."""
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(LATENT_DIM, N_EXPERTS)

    def forward(self, z_ec, temperature=1.0):
        return F.gumbel_softmax(self.router(z_ec), tau=temperature, hard=True)


class MotorCortexMoE(nn.Module):
    """
    Thalamus-gated Mixture-of-Experts MC.
    EC → z_ec → Thalamus → hard routing → selected expert([s_t, g]) → action.
    Each expert has fully independent weights → B training cannot overwrite A's expert.
    """
    def __init__(self):
        super().__init__()
        self.thalamus = ThalamusRouter()
        self.experts  = nn.ModuleList([MCExpert() for _ in range(N_EXPERTS)])

    def init_hidden(self, batch_size, device):
        return [e.init_hidden(batch_size, device) for e in self.experts]

    def step(self, s_t, z_ec_smooth, g, h_mc_list, temperature=1.0):
        weights    = self.thalamus(z_ec_smooth, temperature=temperature)   # (1, N_EXPERTS) one-hot
        new_h_list = []
        actions    = []
        for i, expert in enumerate(self.experts):
            a_i, h_i = expert.step(s_t, g, h_mc_list[i])
            actions.append(a_i)
            new_h_list.append(h_i)
        action = (torch.stack(actions, dim=1) * weights.unsqueeze(-1)).sum(dim=1)
        return action, new_h_list, weights


# ── Env step ──────────────────────────────────────────────────────────────────
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


# ── Single episode ─────────────────────────────────────────────────────────────
def run_episode(ec, cb, mc, device, ctx, rng, train=True,
                use_ec=True, use_cb=True, cb_noec=None):
    """
    use_ec    : whether EC updates its hidden state and provides z_ec to MC
    use_cb    : whether CB hidden state is passed to MC
    cb_noec   : if not None (condition 2), use CB_NoEC for MC's z_cb
                (original CB still used for EC update when use_ec=True)
    """
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

    step_losses = []
    success     = False
    steps       = 0
    first_angle = None

    for t in range(MAX_STEPS):
        d = torch.norm(pos - g_t).item()
        if d < GOAL_RADIUS:
            success = True; break

        # CB hidden for MC
        if use_cb:
            with torch.no_grad() if not train else torch.enable_grad():
                if cb_noec is not None:
                    cb_h = cb_noec.hidden(pos.unsqueeze(0), prev_a)
                else:
                    cb_h = cb.hidden(pos.unsqueeze(0), prev_a, z_t)
        else:
            cb_h = torch.zeros(1, CB_HIDDEN, device=device)

        h_ec_in = h_ec if use_ec else torch.zeros(1, EC_HIDDEN, device=device)
        a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec_in, g_t.unsqueeze(0), h_mc, cb_h)

        s_next, vel_next = env_step_torch(
            pos, vel, a_t.squeeze(0), beta, theta, gamma, g, bx, by, device)

        if train:
            step_losses.append(torch.sum((g_t - s_next) ** 2))
            # condition 2: also train CB_NoEC as a forward model
            if cb_noec is not None:
                pred = cb_noec(pos.unsqueeze(0), a_t.detach())
                step_losses.append(torch.sum((s_next.detach() - pred) ** 2))

        with torch.no_grad():
            if t == 0:
                a_np  = a_t.squeeze(0).detach().cpu().numpy()
                gd    = (g_t - pos).cpu().numpy()
                cross = gd[0]*a_np[1] - gd[1]*a_np[0]
                dot   = a_np[0]*gd[0] + a_np[1]*gd[1]
                first_angle = float(np.degrees(np.arctan2(cross, dot)))

            if use_ec:
                s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                eps   = s_next.detach().unsqueeze(0) - cb(pos.unsqueeze(0), a_t.detach(), z_t)
                z_t, h_ec = ec.step(
                    torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

        prev_a = a_t.detach()
        pos    = s_next.detach()
        vel    = vel_next
        steps += 1

    first_angle = first_angle if first_angle is not None else 0.0
    loss = torch.stack(step_losses).sum() if step_losses else \
           torch.tensor(0.0, device=device, requires_grad=True)
    return loss, float(success), steps, first_angle


# ── Thalamus MoE episode ──────────────────────────────────────────────────────
def run_episode_moe(ec, cb, mc, device, ctx, rng, train=True, temperature=1.0):
    """Episode for MotorCortexMoE — EC provides z_ec to Thalamus router."""
    beta, theta, gamma, g, bx, by, _ = ctx

    start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(start - goal) < 0.5:
        goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    g_t       = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos       = torch.tensor(start, dtype=torch.float32, device=device)
    vel       = torch.zeros(2, device=device)
    h_ec      = ec.init_hidden(1, device)
    h_mc_list = mc.init_hidden(1, device)
    z_t       = torch.zeros(1, LATENT_DIM, device=device)
    z_smooth  = torch.zeros(1, LATENT_DIM, device=device)
    prev_a    = torch.zeros(1, 2, device=device)

    step_losses = []
    success     = False
    steps       = 0
    first_angle = None

    for t in range(MAX_STEPS):
        if torch.norm(pos - g_t).item() < GOAL_RADIUS:
            success = True; break

        z_smooth = EMA_ALPHA * z_smooth + (1 - EMA_ALPHA) * z_t

        a_t, h_mc_list, _ = mc.step(
            pos.unsqueeze(0), z_smooth, g_t.unsqueeze(0), h_mc_list, temperature)

        s_next, vel_next = env_step_torch(
            pos, vel, a_t.squeeze(0), beta, theta, gamma, g, bx, by, device)

        if train:
            step_losses.append(torch.sum((g_t - s_next) ** 2))

        with torch.no_grad():
            if t == 0:
                a_np  = a_t.squeeze(0).detach().cpu().numpy()
                gd    = (g_t - pos).cpu().numpy()
                cross = gd[0]*a_np[1] - gd[1]*a_np[0]
                dot   = a_np[0]*gd[0] + a_np[1]*gd[1]
                first_angle = float(np.degrees(np.arctan2(cross, dot)))

            s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
            eps   = s_next.detach().unsqueeze(0) - cb(pos.unsqueeze(0), a_t.detach(), z_t)
            z_t, h_ec = ec.step(
                torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

        prev_a = a_t.detach()
        pos    = s_next.detach()
        vel    = vel_next
        steps += 1

    first_angle = first_angle if first_angle is not None else 0.0
    loss = torch.stack(step_losses).sum() if step_losses else \
           torch.tensor(0.0, device=device, requires_grad=True)
    return loss, float(success), steps, first_angle


# ── Training loop ─────────────────────────────────────────────────────────────
def run_aba(ec, cb, mc, device, seed=0, use_ec=True, use_cb=True, cb_noec=None):
    is_moe = isinstance(mc, MotorCortexMoE)
    ec.eval(); cb.eval(); mc.train()
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)
    if is_moe:
        # freeze pretrained thalamus router — routing cannot collapse from task loss
        for p in mc.thalamus.parameters(): p.requires_grad_(False)

    # condition 2: train CB_NoEC jointly with MC
    params = list(mc.parameters())
    if cb_noec is not None:
        cb_noec.train()
        params += list(cb_noec.parameters())

    optimizer   = torch.optim.Adam(params, lr=LR_MC)
    rng         = np.random.default_rng(seed)
    total_eps   = N_PHASE_EP * len(PHASES)

    curve = {"ep": [], "success": [], "steps": [], "first_angle": [], "phase": [],
             "probe_a_ep": [], "probe_a_succ": [], "probe_a_phase": [],
             "mc_state": None}
    ep_global = 0

    for phase_idx, ctx in enumerate(PHASES):
        label = PHASE_LABELS[phase_idx]
        print(f"\n  Phase {label} ({N_PHASE_EP} episodes)...")

        for ep in range(1, N_PHASE_EP + 1):
            ep_global += 1
            tau = TAU_START - (TAU_START - TAU_END) * (ep_global / total_eps)
            mc.train()
            if cb_noec is not None: cb_noec.train()

            optimizer.zero_grad()
            if is_moe:
                loss, _, _, _ = run_episode_moe(ec, cb, mc, device, ctx, rng,
                                                train=True, temperature=tau)
            else:
                loss, _, _, _ = run_episode(ec, cb, mc, device, ctx, rng, train=True,
                                            use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            mc.eval()
            if cb_noec is not None: cb_noec.eval()
            with torch.no_grad():
                if is_moe:
                    _, succ, steps, first_angle = run_episode_moe(
                        ec, cb, mc, device, ctx, rng, train=False, temperature=tau)
                else:
                    _, succ, steps, first_angle = run_episode(
                        ec, cb, mc, device, ctx, rng, train=False,
                        use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec)

            curve["ep"].append(ep_global)
            curve["success"].append(float(succ))
            curve["steps"].append(steps)
            curve["first_angle"].append(float(first_angle))
            curve["phase"].append(phase_idx)

            # probe context A retention during B phases (phases 2 and 4)
            if phase_idx in [2, 4]:
                with torch.no_grad():
                    if is_moe:
                        _, probe_succ, _, _ = run_episode_moe(
                            ec, cb, mc, device, CTX_A, rng, train=False, temperature=tau)
                    else:
                        _, probe_succ, _, _ = run_episode(
                            ec, cb, mc, device, CTX_A, rng, train=False,
                            use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec)
                curve["probe_a_ep"].append(ep_global)
                curve["probe_a_succ"].append(float(probe_succ))
                curve["probe_a_phase"].append(phase_idx)

            if ep % 500 == 0:
                recent_acc = np.mean(curve["success"][-100:])
                print(f"    ep {ep:4d} | acc (last 100)={recent_acc:.2f}")

    curve["mc_state"] = {k: v.cpu().clone() for k, v in mc.state_dict().items()}
    return curve


# ── EC lesion eval ─────────────────────────────────────────────────────────────
def run_lesion_eval(ec, cb, mc_state, device, use_ec, use_cb, seed=0):
    """
    Reload trained MC and run eval across all phases with EC lesioned (z_ec=0).
    No weight updates — pure forward pass.
    """
    mc = MotorCortex(use_ec=use_ec, use_cb=use_cb).to(device)
    mc.load_state_dict({k: v.to(device) for k, v in mc_state.items()})
    mc.eval()
    rng = np.random.default_rng(seed)

    curve = {"success": [], "first_angle": [], "phase": []}

    for phase_idx, ctx in enumerate(PHASES):
        for ep in range(N_PHASE_EP):
            with torch.no_grad():
                # use_ec=False zeros out h_ec_in → lesions EC signal to MC
                _, succ, _, first_angle = run_episode(
                    ec, cb, mc, device, ctx, rng, train=False,
                    use_ec=False, use_cb=use_cb)
            curve["success"].append(float(succ))
            curve["first_angle"].append(float(first_angle))
            curve["phase"].append(phase_idx)

    return curve


# ── Fig 5: EC lesion comparison ────────────────────────────────────────────────
def plot_lesion(curves, lesion_curves):
    """
    Compare trained MC+EC with EC lesioned at eval vs MC+CB (never had EC).
    Key test: if lesioned MC+EC << MC+CB, MC was relying on EC.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    SW = 20
    ep_local = np.arange(1, N_PHASE_EP + 1)
    N_PHASES = len(PHASES)
    phase_boundaries = [i * N_PHASE_EP for i in range(1, N_PHASES)]

    def split_phases(arr):
        arr = np.array(arr)
        return [arr[i*N_PHASE_EP:(i+1)*N_PHASE_EP] for i in range(N_PHASES)]

    # conditions to show: MC+CB (idx 1), MC+EC (idx 2), MC+EC lesioned, MC+EC+CB (idx 3), MC+EC+CB lesioned
    plot_specs = [
        ("MC + CB (no EC)",          curves[1],        "steelblue",    "-"),
        ("MC + EC (intact)",          curves[2],        "seagreen",     "-"),
        ("MC + EC (EC lesioned)",     lesion_curves[2], "seagreen",     "--"),
        ("MC + EC + CB (intact)",     curves[3],        "tomato",       "-"),
        ("MC + EC + CB (EC lesioned)",lesion_curves[3], "tomato",       "--"),
    ]

    SW = 100
    fig, axes = plt.subplots(1, N_PHASES, figsize=(22, 4), sharey=True)

    for ax, phase_idx in zip(axes, range(N_PHASES)):
        for label, curve, color, ls in plot_specs:
            y_raw = np.array(split_phases(curve["success"])[phase_idx])
            if len(y_raw) >= SW:
                y_sm = smooth(y_raw, SW)
                ep_x = ep_local[SW-1:]
            else:
                y_sm = y_raw; ep_x = ep_local
            ax.plot(ep_x, y_sm, color=color, ls=ls, lw=1.8,
                    label=label if phase_idx == 0 else "")

        ax.set_title(PHASE_LABELS[phase_idx], fontsize=9)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        if phase_idx == 0:
            ax.set_ylabel("Success rate", fontsize=9)

    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("EC lesion test: does MC+EC rely on z_ec?\n"
                 "(dashed = EC lesioned at eval, solid = intact | "
                 "if dashed << MC+CB → MC relied on EC)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_ec_lesion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_ec_lesion.png")

    # ── Zoomed: A and A2 phases only, first 500 episodes ─────────────────────
    zoom_phases = [1, 3]   # A and A2
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, phase_idx in zip(axes2, zoom_phases):
        for label, curve, color, ls in plot_specs:
            y_raw = np.array(split_phases(curve["success"])[phase_idx])
            if len(y_raw) >= SW:
                y_sm = smooth(y_raw, SW)
                ep_x = ep_local[SW-1:]
            else:
                y_sm = y_raw; ep_x = ep_local
            mask = ep_x <= 500
            ax.plot(ep_x[mask], y_sm[mask], color=color, ls=ls, lw=2,
                    label=label if phase_idx == zoom_phases[0] else "")
        ax.set_title(PHASE_LABELS[phase_idx], fontsize=10)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(1, 500)
        ax.spines[["top", "right"]].set_visible(False)
        if phase_idx == zoom_phases[0]:
            ax.set_ylabel("Success rate", fontsize=9)
    axes2[0].legend(fontsize=8, loc="lower right")
    fig2.suptitle("EC lesion — zoomed: A and A2 phases (ep 1–500)\n"
                  "(dashed = lesioned, solid = intact)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_ec_lesion_zoom.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_ec_lesion_zoom.png")


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
        popt, _ = curve_fit(model, x, y, p0=[float(y[-1]), float(y[0]), 1e-3], maxfev=5000)
        L, y0, k = popt
        x_95 = -np.log(0.05) / k if k > 0 and abs(L - y0) > 1e-6 else None
        return L, y0, k, x_95
    except Exception:
        return None, None, None, None


def plot_all(curves):
    os.makedirs(FIG_DIR, exist_ok=True)
    N_PHASES = len(PHASES)
    N_CONDS  = len(CONDITIONS)
    SW       = 20
    ep_local = np.arange(1, N_PHASE_EP + 1)
    phase_boundaries = [i * N_PHASE_EP for i in range(1, N_PHASES)]
    row_ys   = np.linspace(0.13, 0.90, N_CONDS)[::-1].tolist()

    def split_phases(arr):
        arr = np.array(arr)
        return [arr[i*N_PHASE_EP:(i+1)*N_PHASE_EP] for i in range(N_PHASES)]

    # ── Fig 1: N×3 learning curves ────────────────────────────────────────────
    fig1, axes1 = plt.subplots(N_CONDS, 3, figsize=(22, 4*N_CONDS), sharey="col")

    for row, (cond, color, curve) in enumerate(zip(CONDITIONS, COND_COLORS, curves)):
        succ_phases  = split_phases(curve["success"])
        steps_phases = split_phases(curve["steps"])

        for col, (phases, ylabel, use_smooth) in enumerate(zip(
            [succ_phases, steps_phases, succ_phases],   # placeholder for dist
            ["Success rate", "Steps to goal", "Success rate"],
            [True, True, True],
        )):
            # override col 2 with steps for now — dist not tracked
            if col == 2:
                phases     = steps_phases
                ylabel     = "Steps to goal"
                use_smooth = True

            ax = axes1[row, col]
            for phase_idx, y_raw in enumerate(phases):
                y    = smooth(y_raw, SW) if use_smooth else y_raw
                ep_x = ep_local[SW-1:] if use_smooth else ep_local
                x_offset = phase_idx * N_PHASE_EP
                ax.plot(x_offset + ep_x, y, color=color, lw=1.5, alpha=0.8)

                L, y0, k, x_95 = fit_asymptote(ep_x, y)
                if L is not None and k is not None:
                    y_fit = L + (y0 - L) * np.exp(-k * ep_x)
                    ax.plot(x_offset + ep_x, y_fit, color="black", ls="--", lw=1, zorder=3)

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
            if row == 3:
                ax.set_xlabel("Episode within phase", fontsize=8)

            if ylabel == "Success rate":
                ax.set_ylim(0, 1.05)
            ax.spines[["top", "right"]].set_visible(False)

            # phase labels top row only
            if row == 0:
                y_top = 1.0 if ylabel == "Success rate" else ax.get_ylim()[1]
                for i, lbl in enumerate(PHASE_LABELS):
                    mid = i * N_PHASE_EP + N_PHASE_EP // 2
                    ax.text(mid, y_top, lbl, ha="center", va="bottom",
                            fontsize=6, color="dimgray",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    fig1.suptitle("ABA savings — learning curves across conditions\n"
                  "(rolling mean w=20 | black dashed = exponential fit)",
                  fontsize=12)
    plt.tight_layout()
    for row_y, cond, color in zip(row_ys, CONDITIONS, COND_COLORS):
        fig1.text(0.01, row_y, cond["name"], va="center", ha="left",
                  fontsize=11, fontweight="bold", color=color,
                  rotation=90)
    plt.savefig(f"{FIG_DIR}/savings_all_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved savings_all_learning_curves.png")

    # ── Fig 2: N×2 angle savings overlay ─────────────────────────────────────
    SW_OV = 25
    fig2, axes2 = plt.subplots(N_CONDS, 2, figsize=(14, 4*N_CONDS))

    for row, (cond, color, curve) in enumerate(zip(CONDITIONS, COND_COLORS, curves)):
        angle_phases = split_phases(curve["first_angle"])

        for col, (idx1, idx2, target, title) in enumerate([
            (1, 3, -60, "A vs A2  (rot+60°)"),
            (2, 4,   0, "B vs B2  (rot 0°)"),
        ]):
            ax = axes2[row, col]

            for idx, shade, label in [
                (idx1, 0.4,  PHASE_LABELS[idx1]),
                (idx2, 1.0,  PHASE_LABELS[idx2]),
            ]:
                y_raw = np.array(angle_phases[idx])
                y_sm  = smooth(y_raw, SW_OV)
                ep_sm = ep_local[SW_OV - 1:]
                mask  = ep_sm <= 350
                ax.plot(ep_sm[mask], y_sm[mask], color=color,
                        lw=2, alpha=shade, label=label)

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
                ax.set_ylabel(f"{cond['name']}\nAngle (deg)", fontsize=8)
            if row == 3:
                ax.set_xlabel("Episode within phase", fontsize=8)

    fig2.suptitle("ABA savings — angle savings overlay across conditions\n"
                  f"(rolling mean w={SW_OV} | faded=first exposure | solid=second exposure)",
                  fontsize=12)
    plt.tight_layout()
    for row_y, cond, color in zip(row_ys, CONDITIONS, COND_COLORS):
        fig2.text(0.01, row_y, cond["name"], va="center", ha="left",
                  fontsize=11, fontweight="bold", color=color,
                  rotation=90)
    plt.savefig(f"{FIG_DIR}/savings_all_angle_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved savings_all_angle_overlay.png")

    # ── Fig 3: 4×3 ABA accuracy asymptote (A, B, A2 phases only) ─────────────
    ABA_INDICES = [1, 2, 3]   # A, B, A2
    ABA_LABELS  = [PHASE_LABELS[i] for i in ABA_INDICES]
    SW_ACC      = 20

    fig3, axes3 = plt.subplots(N_CONDS, 3, figsize=(15, 4*N_CONDS), sharey=True)

    for row, (cond, color, curve) in enumerate(zip(CONDITIONS, COND_COLORS, curves)):
        succ_all = split_phases(curve["success"])

        for col, (phase_idx, lbl) in enumerate(zip(ABA_INDICES, ABA_LABELS)):
            ax = axes3[row, col]
            y_raw = np.array(succ_all[phase_idx])
            y_sm  = smooth(y_raw, SW_ACC)
            ep_sm = ep_local[SW_ACC - 1:]

            ax.plot(ep_sm, y_sm, color=color, lw=2, alpha=0.4, zorder=1)

            L, y0, k, x_95 = fit_asymptote(ep_sm, y_sm)
            if L is not None and k is not None:
                y_fit = L + (y0 - L) * np.exp(-k * ep_sm)
                ax.plot(ep_sm, y_fit, color="tomato", ls="--", lw=2,
                        label=f"asymptote={L:.2f}", zorder=3)
            if x_95 is not None and x_95 < N_PHASE_EP:
                ax.axvline(x_95, color="black", ls=":", lw=1.5, zorder=2)
                ax.text(x_95, (L + 0.03) if L is not None else 0.5,
                        f"ep {int(x_95)}", color="black", fontsize=7,
                        ha="center", va="bottom")

            ax.set_xlim(1, N_PHASE_EP)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)

            if row == 0:
                ax.set_title(lbl, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{cond['name']}\nSuccess rate", fontsize=8,
                              color=color, fontweight="bold")
            if row == 3:
                ax.set_xlabel("Episode within phase", fontsize=8)

    fig3.suptitle("ABA accuracy learning curves with asymptote fits — 4 conditions\n"
                  f"(rolling mean w={SW_ACC} | red dashed = exponential fit)",
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_all_acc_asymptote.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved savings_all_acc_asymptote.png")


# ── Fig 4: Forgetting + Savings ───────────────────────────────────────────────
def plot_forgetting_and_savings(curves):
    os.makedirs(FIG_DIR, exist_ok=True)
    SW      = 30
    ep_loc  = np.arange(1, N_PHASE_EP + 1)

    def split_phases(arr):
        arr = np.array(arr)
        return [arr[i*N_PHASE_EP:(i+1)*N_PHASE_EP] for i in range(len(PHASES))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: A-probe during B phase ──────────────────────────────────────────
    ax = axes[0]
    for cond, color, curve in zip(CONDITIONS, COND_COLORS, curves):
        probe_succ = np.array(curve.get("probe_a_succ", []))
        probe_ph   = np.array(curve.get("probe_a_phase", []))
        mask = probe_ph == 2
        if mask.sum() > SW:
            y_sm = smooth(probe_succ[mask], SW)
            ax.plot(np.arange(1, len(y_sm)+1), y_sm, color=color, lw=2,
                    label=cond["name"])
    ax.set_xlabel("Episode within B phase", fontsize=10)
    ax.set_ylabel("Context-A success rate (probe, no update)", fontsize=10)
    ax.set_title("Does B overwrite A?\n(probed on context A throughout B training)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Right: Savings bar chart (x_95: A1 vs A2) ─────────────────────────────
    ax = axes[1]
    x_95_A, x_95_A2 = [], []
    for curve in curves:
        succ_phases = split_phases(curve["success"])
        for phase_idx, store in [(1, x_95_A), (3, x_95_A2)]:
            y_sm = smooth(np.array(succ_phases[phase_idx]), 20)
            ep_sm = ep_loc[19:]
            _, _, k, x95 = fit_asymptote(ep_sm, y_sm)
            store.append(x95 if (x95 is not None and x95 < N_PHASE_EP) else N_PHASE_EP)

    width = 0.35
    xs = np.arange(len(CONDITIONS))
    for i, (cond, color, a, a2) in enumerate(zip(CONDITIONS, COND_COLORS, x_95_A, x_95_A2)):
        ax.bar(i - width/2, a,  width, color=color, alpha=0.4)
        ax.bar(i + width/2, a2, width, color=color, alpha=1.0)
        saving = a - a2
        mid_y  = (a + a2) / 2
        ax.annotate("", xy=(i + width/2, a2), xytext=(i + width/2, a),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.text(i + width/2 + 0.07, mid_y, f"Δ={saving:.0f}", fontsize=8, va="center")

    ax.set_xticks(xs)
    ax.set_xticklabels([c["name"] for c in CONDITIONS], fontsize=8,
                       rotation=15, ha="right")
    ax.set_ylabel("Episodes to 95% asymptote", fontsize=10)
    ax.set_title("Savings: A1 vs A2\n(faded=1st, solid=2nd, arrow=savings)", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="A (1st)"),
                        Patch(facecolor="gray", alpha=1.0, label="A2 (2nd)")], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/savings_forgetting_and_savings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved savings_forgetting_and_savings.png")


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
    data_path = f"{FIG_DIR}/savings_all_data.npz"

    if os.path.exists(data_path) and "--replot" in sys.argv:
        print("Loading saved data — skipping training.")
        d      = np.load(data_path, allow_pickle=True)
        curves = list(d["curves"])
    else:
        curves = []
        for cond in CONDITIONS:
            print(f"\n{'='*50}")
            print(f"Condition: {cond['name']}")
            print(f"{'='*50}")

            mc      = MotorCortex(use_ec=cond["use_ec"], use_cb=cond["use_cb"]).to(device)
            cb_noec = CerebellumCB_NoEC().to(device) if cond["retrain_cb"] else None

            curve = run_aba(ec, cb, mc, device, seed=0,
                            use_ec=cond["use_ec"],
                            use_cb=cond["use_cb"],
                            cb_noec=cb_noec)
            curves.append(curve)

        np.savez(data_path, curves=curves)
        print(f"\nSaved data to {FIG_DIR}/")

    plot_all(curves)
    plot_forgetting_and_savings(curves)

    # EC lesion eval — reload trained MC+EC and MC+EC+CB, run with z_ec=0
    print("\nRunning EC lesion evals...")
    lesion_curves = [None] * len(CONDITIONS)
    for i, cond in enumerate(CONDITIONS):
        if cond["use_ec"] and curves[i].get("mc_state") is not None:
            print(f"  Lesioning: {cond['name']}")
            lesion_curves[i] = run_lesion_eval(
                ec, cb, curves[i]["mc_state"], device,
                use_ec=cond["use_ec"], use_cb=cond["use_cb"], seed=0)
    plot_lesion(curves, lesion_curves)
    print("Done.")


if __name__ == "__main__":
    main()

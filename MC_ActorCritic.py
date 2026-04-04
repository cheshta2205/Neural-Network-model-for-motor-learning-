"""
mc_ec_cb_babab.py

Single-condition BABAB motor-learning script:
    MC + EC + CB

Architecture
------------
- MC policy input: s, g, z_ec, z_cb
- EC is loaded from checkpoint and frozen
- CB is loaded from checkpoint and frozen
- MC is the only trainable module

Task
----
BABAB phases:
    B1 -> A1 -> B2 -> A2 -> B3

Success metric
--------------
Episode success = 1 if agent reaches goal within GOAL_RADIUS during episode,
otherwise 0.

Plots
-----
For each seed:
- global success curve
- global return curve
- A1 vs A2 success
- ABA success grid
- A1 vs A2 angle overlay
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM   = 32
EC_HIDDEN    = 256
CB_HIDDEN    = 256
EC_INPUT     = 8
CB_PROJ_DIM  = 32
MC_HIDDEN    = 256

MAX_STEPS    = 40
N_PHASE_EP   = 700
GOAL_RADIUS  = 0.35
GRID_SIZE    = 4.0
DT           = 1.0

LR_MC          = 3e-4
WEIGHT_DECAY   = 0.0

GAMMA              = 0.97
ENTROPY_WEIGHT     = 0.002
ACTION_STD_MIN     = -2.5
ACTION_STD_MAX     = -0.2
ACTION_L2_WEIGHT   = 1e-4
BASELINE_ALPHA     = 0.05   # EMA decay for running return baseline

SUCCESS_BONUS      = 5.0
DIST_REWARD_SCALE  = 1.0

CKPT    = "checkpoints/ectocb_8ctx_seq80_z32.pt"
FIG_DIR = "figs/mc_ec_cb_babab_actorcritic"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Contexts / phases ─────────────────────────────────────────────────────────
CTX_A = (1.0, np.pi / 3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B = (1.0, 0.0,      +0.4, 0.0, 0.0, 0.0, "B: curl+")

PHASES       = [CTX_B, CTX_A, CTX_B, CTX_A, CTX_B]
PHASE_LABELS = ["B1: curl+", "A1: rot+60°", "B2: curl+", "A2: rot+60°", "B3: curl+"]

A1_IDX = 1
B2_IDX = 2
A2_IDX = 3

A_IDX = A1_IDX
B_IDX = B2_IDX


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_rotation_matrix(theta, device):
    cos_t = torch.tensor(float(np.cos(theta)), dtype=torch.float32, device=device)
    sin_t = torch.tensor(float(np.sin(theta)), dtype=torch.float32, device=device)
    return torch.stack([
        torch.stack([cos_t, -sin_t]),
        torch.stack([sin_t,  cos_t]),
    ])


def env_step_torch(pos, vel, action, beta, theta, gamma, g, bx, by, device):
    R    = get_rotation_matrix(theta, device)
    curl = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32, device=device)
    b_v  = torch.tensor([float(bx), float(by)], dtype=torch.float32, device=device)

    F = beta * (R @ action) + gamma * (curl @ vel) + g * b_v
    vel_new = vel + F
    pos_new = torch.clamp(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new.detach()


def compute_reward(s_next, g_t, reached):
    dist = torch.norm(s_next - g_t)
    reward = -DIST_REWARD_SCALE * dist
    if reached:
        reward = reward + SUCCESS_BONUS
    return reward


# ── Models ────────────────────────────────────────────────────────────────────
class EntorhinalCortex(nn.Module):
    def __init__(self):
        super().__init__()
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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + LATENT_DIM, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, 2)
        )

    def forward(self, s, a, z):
        return self.net(torch.cat([s, a, z], dim=-1))

    def hidden(self, s, a, z):
        x = self.net[0](torch.cat([s, a, z], dim=-1))
        x = self.net[1](x)
        x = self.net[2](x)
        x = self.net[3](x)
        return x


class MotorCortex(nn.Module):
    """
    MC policy:
        pi(a | s, g, z_ec, z_cb)
    """
    def __init__(self):
        super().__init__()
        input_dim = 2 + 2 + LATENT_DIM + CB_PROJ_DIM

        self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.cb_proj = nn.Linear(CB_HIDDEN, CB_PROJ_DIM, bias=False)

        self.cell = nn.GRUCell(input_dim, MC_HIDDEN)
        self.mu_head = nn.Linear(MC_HIDDEN, 2)
        self.log_std_head = nn.Linear(MC_HIDDEN, 2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, h_ec, g_t, z_cb_t, h_mc):
        z_ec = torch.tanh(self.ec_readout(h_ec))
        z_cb = self.cb_proj(z_cb_t)

        x = torch.cat([s_t, g_t, z_ec, z_cb], dim=-1)
        h_new = self.cell(x, h_mc)

        mu = self.mu_head(h_new)
        log_std = torch.clamp(self.log_std_head(h_new), ACTION_STD_MIN, ACTION_STD_MAX)
        return mu, log_std, h_new

    def sample_action(self, s_t, h_ec, g_t, z_cb_t, h_mc):
        mu, log_std, h_new = self.step(s_t, h_ec, g_t, z_cb_t, h_mc)

        if torch.isnan(mu).any() or torch.isnan(log_std).any():
            raise RuntimeError(f"NaN in policy outputs: mu={mu}, log_std={log_std}")

        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)

        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob = log_prob - torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return a, log_prob, entropy, h_new

    def greedy_action(self, s_t, h_ec, g_t, z_cb_t, h_mc):
        mu, _, h_new = self.step(s_t, h_ec, g_t, z_cb_t, h_mc)
        a = torch.tanh(mu)
        return a, h_new


# ── Single episode ────────────────────────────────────────────────────────────
def run_episode(ec, cb_frozen, mc, device, ctx, rng, train=True, baseline=0.0):
    beta, theta, gamma, g_dyn, bx, by, _ = ctx

    start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(start - goal) < 0.5:
        goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    g_t  = torch.tensor(goal, dtype=torch.float32, device=device)
    pos  = torch.tensor(start, dtype=torch.float32, device=device)
    vel  = torch.zeros(2, dtype=torch.float32, device=device)

    h_ec = ec.init_hidden(1, device)
    h_mc = mc.init_hidden(1, device)

    z_t = torch.zeros(1, LATENT_DIM, device=device)
    prev_a = torch.zeros(1, 2, device=device)

    rewards = []
    log_probs = []
    entropies = []
    actions = []

    dist_seq = []
    angle_seq = []

    success = False
    steps = 0
    first_angle = None
    init_dist = torch.norm(pos - g_t).item()

    for t in range(MAX_STEPS):
        d = torch.norm(pos - g_t).item()
        dist_seq.append(d)

        if d < GOAL_RADIUS:
            success = True
            break

        with torch.no_grad():
            z_cb_t = cb_frozen.hidden(pos.unsqueeze(0), prev_a, z_t)

        if train:
            a_t, log_prob_t, entropy_t, h_mc = mc.sample_action(
                pos.unsqueeze(0), h_ec, g_t.unsqueeze(0), z_cb_t, h_mc
            )
            log_probs.append(log_prob_t.squeeze(0))
            entropies.append(entropy_t.squeeze(0))
        else:
            a_t, h_mc = mc.greedy_action(
                pos.unsqueeze(0), h_ec, g_t.unsqueeze(0), z_cb_t, h_mc
            )

        s_next, vel_next = env_step_torch(
            pos, vel, a_t.squeeze(0), beta, theta, gamma, g_dyn, bx, by, device
        )

        reached = bool(torch.norm(s_next - g_t).item() < GOAL_RADIUS)
        r_t = compute_reward(s_next, g_t, reached)

        rewards.append(r_t)
        actions.append(a_t.squeeze(0))

        with torch.no_grad():
            a_np  = a_t.squeeze(0).detach().cpu().numpy()
            gd    = (g_t - pos).cpu().numpy()
            cross = gd[0] * a_np[1] - gd[1] * a_np[0]
            dot   = a_np[0] * gd[0] + a_np[1] * gd[1]
            alpha = float(np.degrees(np.arctan2(cross, dot)))
            angle_seq.append(alpha)
            if t == 0:
                first_angle = alpha

        # EC update using CB prediction error
        with torch.no_grad():
            s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
            cb_pred = cb_frozen(pos.unsqueeze(0), a_t.detach(), z_t)
            eps = s_next.detach().unsqueeze(0) - cb_pred
            z_t, h_ec = ec.step(
                torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1),
                h_ec
            )

        prev_a = a_t.detach()
        pos = s_next.detach()
        vel = vel_next
        steps += 1

        if reached:
            success = True
            break

    # discounted returns
    returns = []
    G = 0.0
    for r_t in reversed(rewards):
        G = float(r_t.detach().cpu().item()) + GAMMA * G
        returns.append(G)
    returns = list(reversed(returns))

    # Running-baseline REINFORCE: adv = G_t - baseline (cross-episode signal preserved)
    ep_G0 = returns[0] if returns else 0.0

    if train and len(returns) > 0:
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advs = returns_t - baseline   # centre by running mean, no within-ep normalisation

        mc_terms = []
        for adv, lp, ent, a in zip(advs, log_probs, entropies, actions):
            actor_term = -adv * lp
            entropy_term = -ENTROPY_WEIGHT * ent
            action_pen = ACTION_L2_WEIGHT * torch.sum(a ** 2)
            mc_terms.append(actor_term + entropy_term + action_pen)

        mc_loss = torch.stack(mc_terms).mean()
    else:
        mc_loss = torch.tensor(0.0, device=device, requires_grad=True)

    ep_return = float(sum(float(r.detach().cpu().item()) for r in rewards))
    dist_norm = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
    mean_angle = float(np.mean(angle_seq)) if angle_seq else 0.0
    first_angle = first_angle if first_angle is not None else 0.0

    return {
        "mc_loss": mc_loss,
        "success": float(success),
        "steps": steps,
        "dist_norm": dist_norm,
        "angle": mean_angle,
        "first_angle": first_angle,
        "ep_return": ep_return,
        "G0": ep_G0,
    }


# ── Training loop ─────────────────────────────────────────────────────────────
def run_babab(seed=0):
    device = DEVICE
    rng = np.random.default_rng(seed)

    ckpt = torch.load(CKPT, map_location=device)

    ec = EntorhinalCortex().to(device)
    cb_frozen = CerebellumCB().to(device)
    ec.load_state_dict(ckpt["ec"])
    cb_frozen.load_state_dict(ckpt["cb"])

    ec.eval()
    cb_frozen.eval()

    for p in ec.parameters():
        p.requires_grad_(False)
    for p in cb_frozen.parameters():
        p.requires_grad_(False)

    mc = MotorCortex().to(device)
    mc_optimizer = torch.optim.Adam(mc.parameters(), lr=LR_MC, weight_decay=WEIGHT_DECAY)

    curve = {
        "ep": [],
        "phase": [],
        "success": [],
        "steps": [],
        "dist_norm": [],
        "angle": [],
        "first_angle": [],
        "return": [],
        "mc_loss": [],
    }

    global_ep = 0
    baseline = 0.0   # running EMA of episode return G0

    for phase_idx, ctx in enumerate(PHASES):
        print(f"\nPhase {PHASE_LABELS[phase_idx]} ({N_PHASE_EP} episodes)...")

        for local_ep in range(1, N_PHASE_EP + 1):
            global_ep += 1

            # training rollout
            mc.train()
            out = run_episode(ec, cb_frozen, mc, device, ctx, rng, train=True, baseline=baseline)

            if torch.isnan(out["mc_loss"]):
                raise RuntimeError("mc_loss became NaN")

            loss_val = float(out["mc_loss"].detach().item())
            mc_optimizer.zero_grad()
            out["mc_loss"].backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            mc_optimizer.step()

            # update running baseline with this episode's return
            baseline = (1 - BASELINE_ALPHA) * baseline + BASELINE_ALPHA * out["G0"]

            # deterministic eval rollout
            mc.eval()
            with torch.no_grad():
                eval_out = run_episode(ec, cb_frozen, mc, device, ctx, rng, train=False)

            curve["ep"].append(global_ep)
            curve["phase"].append(phase_idx)
            curve["success"].append(float(eval_out["success"]))
            curve["steps"].append(float(eval_out["steps"]))
            curve["dist_norm"].append(float(eval_out["dist_norm"]))
            curve["angle"].append(float(eval_out["angle"]))
            curve["first_angle"].append(float(eval_out["first_angle"]))
            curve["return"].append(float(eval_out["ep_return"]))
            curve["mc_loss"].append(loss_val)

            if local_ep % 100 == 0:
                print(
                    f"  ep {local_ep:4d} | "
                    f"succ(last100)={np.mean(curve['success'][-100:]):.2f} | "
                    f"ret(last100)={np.mean(curve['return'][-100:]):.2f} | "
                    f"loss(last100)={np.mean(curve['mc_loss'][-100:]):.4f} | "
                    f"baseline={baseline:.3f}"
                )

    return curve


# ── Plot helpers ──────────────────────────────────────────────────────────────
SW = 60
SW_ACC = 60
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


def plot_seed(seed, curve, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    # 1) global success curve
    fig, ax = plt.subplots(figsize=(12, 4))
    succ = np.array(curve["success"])
    y_sm = smooth(succ)
    ep_g = np.arange(1, len(succ) + 1)
    ax.plot(ep_g[SW - 1:], y_sm, lw=1.7)
    for xb in [i * N_PHASE_EP for i in range(1, len(PHASES))]:
        ax.axvline(xb, color="gray", ls="--", lw=0.8)
    for i, lbl in enumerate(PHASE_LABELS):
        ax.text((i + 0.5) * N_PHASE_EP, 1.02, lbl, ha="center",
                transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Episode (global)")
    ax.set_ylabel("Success")
    ax.set_title(f"MC+EC+CB BABAB success curve (seed {seed})")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "learning_curve_success.png"), dpi=150)
    plt.close()

    # 2) global return curve
    fig, ax = plt.subplots(figsize=(12, 4))
    ret = np.array(curve["return"])
    y_sm = smooth(ret)
    ep_g = np.arange(1, len(ret) + 1)
    ax.plot(ep_g[SW - 1:], y_sm, lw=1.7)
    for xb in [i * N_PHASE_EP for i in range(1, len(PHASES))]:
        ax.axvline(xb, color="gray", ls="--", lw=0.8)
    for i, lbl in enumerate(PHASE_LABELS):
        ax.text((i + 0.5) * N_PHASE_EP, 1.02, lbl, ha="center",
                transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_xlabel("Episode (global)")
    ax.set_ylabel("Return")
    ax.set_title(f"MC+EC+CB BABAB return curve (seed {seed})")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "learning_curve_return.png"), dpi=150)
    plt.close()

        # ── Global distance curve (VERY IMPORTANT) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))

    dist = np.array(curve["dist_norm"])   # already stored in your code
    y_sm = smooth(dist)
    ep_g = np.arange(1, len(dist) + 1)

    ax.plot(ep_g[SW - 1:], y_sm, lw=1.7)

    # phase boundaries
    for xb in [i * N_PHASE_EP for i in range(1, len(PHASES))]:
        ax.axvline(xb, color="gray", ls="--", lw=0.8)

    # phase labels
    for i, lbl in enumerate(PHASE_LABELS):
        ax.text((i + 0.5) * N_PHASE_EP, 1.02, lbl,
                ha="center",
                transform=ax.get_xaxis_transform(),
                fontsize=8)

    ax.set_xlabel("Episode (global)")
    ax.set_ylabel("Normalized distance")
    ax.set_title(f"MC+EC+CB BABAB distance curve (seed {seed})")

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "learning_curve_distance.png"), dpi=150)
    plt.close()

    # 3) training loss curve
    fig, ax = plt.subplots(figsize=(12, 4))
    loss_arr = np.array(curve["mc_loss"])
    y_sm = smooth(loss_arr)
    ep_g = np.arange(1, len(loss_arr) + 1)
    ax.plot(ep_g[SW - 1:], y_sm, lw=1.7, color="crimson")
    for xb in [i * N_PHASE_EP for i in range(1, len(PHASES))]:
        ax.axvline(xb, color="gray", ls="--", lw=0.8)
    for i, lbl in enumerate(PHASE_LABELS):
        ax.text((i + 0.5) * N_PHASE_EP, 1.02, lbl, ha="center",
                transform=ax.get_xaxis_transform(), fontsize=8)
    ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("Episode (global)")
    ax.set_ylabel("MC loss (training)")
    ax.set_title(f"MC+EC+CB training loss (seed {seed})")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "training_loss.png"), dpi=150)
    plt.close()

    # 4) A1 vs A2 success
    phases = split_phases(curve["success"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for ph_idx, col, lbl in [(A1_IDX, "steelblue", "A1"), (A2_IDX, "darkorange", "A2")]:
        y_sm = smooth(np.array(phases[ph_idx]))
        ep_sm = ep_local[SW - 1:]
        ax.plot(ep_sm, y_sm, lw=2, color=col, label=lbl)
    ax.set_xlim(1, N_PHASE_EP)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Episode within phase")
    ax.set_ylabel("Success")
    ax.set_title(f"MC+EC+CB A1 vs A2 success (seed {seed})")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "a1_vs_a2_success.png"), dpi=150)
    plt.close()

    # 4) ABA success grid
    ABA_PHASE_INDICES = [A_IDX, B_IDX, A2_IDX]
    ABA_PHASE_LABELS  = [PHASE_LABELS[i] for i in ABA_PHASE_INDICES]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), squeeze=False)
    succ_phases = split_phases(curve["success"])
    for col, ph_idx in enumerate(ABA_PHASE_INDICES):
        ax = axes[0, col]
        y_raw = np.array(succ_phases[ph_idx])
        y_sm = smooth(y_raw, SW_ACC)
        ep_sm = ep_local[SW_ACC - 1:]
        ax.plot(ep_sm, y_sm, lw=1.8)

        y_sm_fit = smooth(y_raw, SW)
        ep_sm_fit = ep_local[SW - 1:]
        L, y0, k, x95 = fit_asymptote(ep_sm_fit, y_sm_fit)
        if L is not None:
            ax.plot(ep_sm_fit, exp_model(ep_sm_fit, L, y0, k), lw=1.2, ls="--", alpha=0.7)
            if x95 is not None and x95 < N_PHASE_EP:
                ax.axvline(x95, lw=1, ls=":", alpha=0.6)
                ax.text(x95 + 20, 0.1, f"x95={x95:.0f}", fontsize=7)

        ax.set_ylim(0, 1.1)
        ax.set_xlim(ep_sm[0], ep_sm[-1])
        ax.set_title(ABA_PHASE_LABELS[col], fontsize=9)
        ax.set_xlabel("Episode within phase", fontsize=8)
        if col == 0:
            ax.set_ylabel("Success", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"MC+EC+CB ABA success grid (seed {seed})", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "aba_grid_success.png"), dpi=150)
    plt.close()

    # 5) A1 vs A2 angle overlay
    angle_phases = split_phases(curve["angle"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for ph_idx, col, lbl in [(A1_IDX, "steelblue", "A1"), (A2_IDX, "darkorange", "A2")]:
        y_sm = smooth(np.array(angle_phases[ph_idx]))
        ep_sm = ep_local[SW - 1:]
        mask = ep_sm <= 400
        ax.plot(ep_sm[mask], y_sm[mask], color=col, lw=2, label=lbl)
    ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° target")
    ax.set_xlim(1, 400)
    ax.set_ylim(-110, 60)
    ax.set_xlabel("Episode within phase")
    ax.set_ylabel("Mean trajectory angle (°)")
    ax.set_title(f"MC+EC+CB A1 vs A2 angle (seed {seed})")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "a1_vs_a2_angle.png"), dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    os.makedirs(FIG_DIR, exist_ok=True)

    for seed in range(2):
        print(f"\n{'=' * 70}")
        print(f"Seed {seed}")
        print(f"{'=' * 70}")

        seed_dir = os.path.join(FIG_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        curve_path = os.path.join(seed_dir, "curve.npz")
        if os.path.exists(curve_path):
            print("Cached run found, loading.")
            d = np.load(curve_path, allow_pickle=True)
            curve = d["curve"].item()
        else:
            curve = run_babab(seed=seed)
            np.savez(curve_path, curve=curve)

        plot_seed(seed, curve, seed_dir)

    print(f"\nDone. Results saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
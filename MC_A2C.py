"""
MC_A2C.py

Single-condition BABAB motor-learning: MC + EC + CB
Uses proper Advantage Actor-Critic (A2C):
    - Actor  : stochastic policy π(a | s, g, z_ec, z_cb)
    - Critic : value head V(h_mc) — shares MC's GRU hidden state

Loss per episode
----------------
    L = L_actor + CRITIC_W * L_critic - ENTROPY_W * L_entropy

Advantage
---------
    A_t = G_t - V(s_t)
    G_t = discounted sum of step rewards from t onwards
    V(s_t) = critic prediction of G_t

Diagnostic plots
----------------
    actor_loss, critic_loss, advantage_std, return, success
    → tells you immediately if A2C is converging
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Normal
from scipy.optimize import curve_fit


# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM   = 32
EC_HIDDEN    = 256
CB_HIDDEN    = 256
EC_INPUT     = 8
CB_PROJ_DIM  = 32
MC_HIDDEN    = 256

MAX_STEPS    = 40
N_PHASE_EP   = 4000
GOAL_RADIUS  = 0.8
GRID_SIZE    = 4.0
DT           = 1.0

LR_MC            = 3e-4
GAMMA            = 0.97
ENTROPY_W        = 0.01
ACTION_STD_MIN   = -2.5
ACTION_STD_MAX   = -0.2
CRITIC_W         = 1.0    # weight on critic loss
BATCH_EPS        = 16     # accumulate gradients over this many episodes

SUCCESS_BONUS    = 5.0

CKPT    = "checkpoints/ectocb_8ctx_seq80_z32.pt"
FIG_DIR = "figs/actor_critic_mc"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ── Contexts / phases ─────────────────────────────────────────────────────────
CTX_A = (1.0, np.pi / 3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B = (1.0, 0.0,      +0.4, 0.0, 0.0, 0.0, "B: curl+")

PHASES       = [CTX_B, CTX_A, CTX_B, CTX_A, CTX_B]
PHASE_LABELS = ["B1: curl+", "A1: rot+60°", "B2: curl+", "A2: rot+60°", "B3: curl+"]
A1_IDX, B2_IDX, A2_IDX = 1, 2, 3
A_IDX, B_IDX = A1_IDX, B2_IDX


# ── Helpers (numpy for speed) ────────────────────────────────────────────────
def env_step_np(pos, vel, action, beta, theta, gamma_curl, g_dyn, bx, by):
    """Environment step in pure numpy — no torch overhead."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    curl = np.array([[0., -1.], [1., 0.]], dtype=np.float32)
    b_v  = np.array([bx, by], dtype=np.float32)
    F       = beta * (R @ action) + gamma_curl * (curl @ vel) + g_dyn * b_v
    vel_new = vel + F
    pos_new = np.clip(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new


# ── Models ────────────────────────────────────────────────────────────────────
class EntorhinalCortex(nn.Module):
    def __init__(self):
        super().__init__()
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
    Actor-Critic MC:
        Actor  : π(a | s, g, z_ec, z_cb) — stochastic Gaussian policy
        Critic : V(h_mc) — shares GRU hidden state with actor
    """
    def __init__(self):
        super().__init__()
        input_dim = 2 + 2 + LATENT_DIM + CB_PROJ_DIM

        self.ec_readout  = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.cb_proj     = nn.Linear(CB_HIDDEN, CB_PROJ_DIM, bias=False)
        self.cell        = nn.GRUCell(input_dim, MC_HIDDEN)
        self.mu_head     = nn.Linear(MC_HIDDEN, 2)
        self.log_std_head = nn.Linear(MC_HIDDEN, 2)
        self.value_head  = nn.Linear(MC_HIDDEN, 1)   # ← critic

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def _forward_hidden(self, s_t, g_t, z_ec, z_cb, h_mc):
        """Shared GRU step given pre-computed z_ec and z_cb."""
        x     = torch.cat([s_t, g_t, z_ec, z_cb], dim=-1)
        h_new = self.cell(x, h_mc)
        mu      = self.mu_head(h_new)
        log_std = torch.clamp(self.log_std_head(h_new), ACTION_STD_MIN, ACTION_STD_MAX)
        V       = self.value_head(h_new)          # critic output
        return mu, log_std, V, h_new

    def step(self, s_t, h_ec, g_t, z_cb_t, h_mc):
        """Live episode step — computes z_ec from EC hidden state."""
        z_ec = torch.tanh(self.ec_readout(h_ec))
        z_cb = self.cb_proj(z_cb_t)
        return self._forward_hidden(s_t, g_t, z_ec, z_cb, h_mc)

    def sample_action(self, s_t, h_ec, g_t, z_cb_t, h_mc):
        mu, log_std, V, h_new = self.step(s_t, h_ec, g_t, z_cb_t, h_mc)
        std  = log_std.exp()
        dist = Normal(mu, std)
        u    = dist.rsample()
        a    = torch.tanh(u)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return a, log_prob, entropy, V, h_new

    def greedy_action(self, s_t, h_ec, g_t, z_cb_t, h_mc):
        mu, _, _, h_new = self.step(s_t, h_ec, g_t, z_cb_t, h_mc)
        return torch.tanh(mu), h_new


# ── Single episode ────────────────────────────────────────────────────────────
def run_episode(ec, cb_frozen, mc, device, ctx, rng, train=True):
    """
    Runs one episode. Env dynamics in numpy, policy forward passes in torch.
    """
    beta, theta, gamma_curl, g_dyn, bx, by, _ = ctx

    pos_np = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal_np = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(pos_np - goal_np) < 0.5:
        goal_np = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    vel_np   = np.zeros(2, dtype=np.float32)
    g_t      = torch.tensor(goal_np, dtype=torch.float32, device=device)
    init_dist = float(np.linalg.norm(pos_np - goal_np))

    h_ec   = ec.init_hidden(1, device)
    h_mc   = mc.init_hidden(1, device)
    z_t    = torch.zeros(1, LATENT_DIM, device=device)
    prev_a = torch.zeros(1, 2, device=device)

    rewards    = []
    log_probs  = []
    entropies  = []
    values     = []
    dist_seq   = []
    angle_seq  = []

    success     = False
    steps       = 0
    first_angle = None

    for t in range(MAX_STEPS):
        d = float(np.linalg.norm(pos_np - goal_np))
        dist_seq.append(d)
        if d < GOAL_RADIUS:
            success = True
            break

        pos_t = torch.tensor(pos_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            z_cb_t = cb_frozen.hidden(pos_t.unsqueeze(0), prev_a, z_t)

        if train:
            a_t, log_prob_t, entropy_t, V_t, h_mc = mc.sample_action(
                pos_t.unsqueeze(0), h_ec, g_t.unsqueeze(0), z_cb_t, h_mc
            )
            log_probs.append(log_prob_t.squeeze(0))
            entropies.append(entropy_t.squeeze(0))
            values.append(V_t.squeeze(0).squeeze(0))
        else:
            with torch.no_grad():
                a_t, h_mc = mc.greedy_action(
                    pos_t.unsqueeze(0), h_ec, g_t.unsqueeze(0), z_cb_t, h_mc
                )

        a_np = a_t.squeeze(0).detach().cpu().numpy()
        s_next_np, vel_new_np = env_step_np(
            pos_np, vel_np, a_np, beta, theta, gamma_curl, g_dyn, bx, by
        )

        reached = bool(np.linalg.norm(s_next_np - goal_np) < GOAL_RADIUS)
        # Negative distance reward (scaled) + success bonus
        r = -0.1 * float(np.linalg.norm(s_next_np - goal_np))
        if reached:
            r += SUCCESS_BONUS
        rewards.append(r)

        # Angle diagnostic
        gd    = goal_np - pos_np
        cross = gd[0] * a_np[1] - gd[1] * a_np[0]
        dot   = a_np[0] * gd[0] + a_np[1] * gd[1]
        alpha = float(np.degrees(np.arctan2(cross, dot)))
        angle_seq.append(alpha)
        if t == 0:
            first_angle = alpha

        # EC update
        with torch.no_grad():
            s_next_t = torch.tensor(s_next_np, dtype=torch.float32, device=device)
            s_tm1    = torch.zeros(1, 2, device=device) if t == 0 else pos_t.unsqueeze(0)
            cb_pred  = cb_frozen(pos_t.unsqueeze(0), a_t.detach(), z_t)
            eps      = s_next_t.unsqueeze(0) - cb_pred
            z_t, h_ec = ec.step(
                torch.cat([s_tm1, prev_a, pos_t.unsqueeze(0), eps], dim=-1), h_ec
            )

        prev_a  = a_t.detach()
        pos_np  = s_next_np
        vel_np  = vel_new_np
        steps  += 1

        if reached:
            success = True
            break

    # ── Compute discounted returns ────────────────────────────────────────────
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.append(G)
    returns = list(reversed(returns))

    if train and len(returns) > 0:
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        values_t  = torch.stack(values)

        # Advantage: normalized for stability
        advantages = returns_t - values_t.detach()
        if len(advantages) > 1:
            std = advantages.std()
            if std > 1e-8:
                advantages = (advantages - advantages.mean()) / (std + 1e-8)

        log_probs_t = torch.stack(log_probs)
        L_actor  = -(advantages * log_probs_t).mean()
        L_critic = F.mse_loss(values_t, returns_t)
        entropies_t = torch.stack(entropies)
        L_entropy   = entropies_t.mean()

        mc_loss = L_actor + CRITIC_W * L_critic - ENTROPY_W * L_entropy
        adv_std = float(advantages.std().item()) if len(advantages) > 1 else 0.0
    else:
        mc_loss   = torch.tensor(0.0, device=device, requires_grad=True)
        L_actor   = mc_loss
        L_critic  = mc_loss
        L_entropy = mc_loss
        adv_std   = 0.0

    ep_return   = float(sum(rewards))
    dist_norm   = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
    mean_angle  = float(np.mean(angle_seq)) if angle_seq else 0.0
    first_angle = first_angle if first_angle is not None else 0.0

    return {
        "mc_loss":   mc_loss,
        "L_actor":   float(L_actor.detach().item()) if train else 0.0,
        "L_critic":  float(L_critic.detach().item()) if train else 0.0,
        "adv_std":   adv_std,
        "success":   float(success),
        "steps":     steps,
        "dist_norm": dist_norm,
        "angle":     mean_angle,
        "first_angle": first_angle,
        "ep_return": ep_return,
        "G0":        returns[0] if returns else 0.0,
    }


# ── Training loop ─────────────────────────────────────────────────────────────
def run_babab(seed=0):
    device = DEVICE
    rng    = np.random.default_rng(seed)

    ckpt = torch.load(CKPT, map_location=device)
    ec         = EntorhinalCortex().to(device)
    cb_frozen  = CerebellumCB().to(device)
    ec.load_state_dict(ckpt["ec"])
    cb_frozen.load_state_dict(ckpt["cb"])
    ec.eval();  cb_frozen.eval()
    for p in ec.parameters():       p.requires_grad_(False)
    for p in cb_frozen.parameters(): p.requires_grad_(False)

    mc           = MotorCortex().to(device)
    mc_optimizer = torch.optim.Adam(mc.parameters(), lr=LR_MC)

    curve = {
        "ep": [], "phase": [], "success": [], "steps": [],
        "dist_norm": [], "angle": [], "first_angle": [], "return": [],
        "L_actor": [], "L_critic": [], "L_total": [], "adv_std": [],
    }

    global_ep = 0

    for phase_idx, ctx in enumerate(PHASES):
        print(f"\nPhase {PHASE_LABELS[phase_idx]} ({N_PHASE_EP} episodes)...")

        mc_optimizer.zero_grad()
        for local_ep in range(1, N_PHASE_EP + 1):
            global_ep += 1

            # ── Train rollout ─────────────────────────────────────────────────
            mc.train()
            out = run_episode(ec, cb_frozen, mc, device, ctx, rng, train=True)

            # Accumulate gradients, update every BATCH_EPS episodes
            (out["mc_loss"] / BATCH_EPS).backward()
            if local_ep % BATCH_EPS == 0:
                torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
                mc_optimizer.step()
                mc_optimizer.zero_grad()

            # Log from training episode
            curve["ep"].append(global_ep)
            curve["phase"].append(phase_idx)
            curve["success"].append(float(out["success"]))
            curve["steps"].append(float(out["steps"]))
            curve["dist_norm"].append(float(out["dist_norm"]))
            curve["angle"].append(float(out["angle"]))
            curve["first_angle"].append(float(out["first_angle"]))
            curve["return"].append(float(out["ep_return"]))
            curve["L_actor"].append(out["L_actor"])
            curve["L_critic"].append(out["L_critic"])
            curve["L_total"].append(float(out["mc_loss"].detach().item()))
            curve["adv_std"].append(out["adv_std"])

            if local_ep % 50 == 0:
                w = min(50, len(curve['success']))
                print(
                    f"  ep {local_ep:4d} | "
                    f"succ={np.mean(curve['success'][-w:]):.2f} | "
                    f"ret={np.mean(curve['return'][-w:]):.1f} | "
                    f"L_act={np.mean(curve['L_actor'][-w:]):.4f} | "
                    f"L_crit={np.mean(curve['L_critic'][-w:]):.4f} | "
                    f"adv_std={np.mean(curve['adv_std'][-w:]):.3f}",
                    flush=True
                )

    return curve


# ── Plotting ──────────────────────────────────────────────────────────────────
SW     = 5
SW_ACC = 5
ep_local = np.arange(1, N_PHASE_EP + 1)


def smooth(x, w=SW):
    x = np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x


def split_phases(arr):
    arr = np.array(arr)
    return [arr[i * N_PHASE_EP:(i + 1) * N_PHASE_EP] for i in range(len(PHASES))]


def exp_model(xv, L, y0, k):
    return L + (y0 - L) * np.exp(-k * xv)


def fit_asymptote(x, y):
    try:
        x, y = np.asarray(x, float), np.asarray(y, float)
        popt, _ = curve_fit(exp_model, x, y,
                            p0=[float(y[-1]), float(y[0]), 1e-3], maxfev=5000)
        L, y0, k = popt
        x95 = -np.log(0.05) / k if (k > 0 and abs(L - y0) > 1e-6) else None
        return L, y0, k, x95
    except Exception:
        return None, None, None, None


def phase_boundaries():
    return [i * N_PHASE_EP for i in range(1, len(PHASES))]


def add_phase_labels(ax, total_eps):
    for xb in phase_boundaries():
        ax.axvline(xb, color="gray", ls="--", lw=0.8)
    for i, lbl in enumerate(PHASE_LABELS):
        ax.text((i + 0.5) * N_PHASE_EP, 1.02, lbl, ha="center",
                transform=ax.get_xaxis_transform(), fontsize=7)


def plot_seed(seed, curve, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    ep_g = np.arange(1, len(curve["success"]) + 1)

    # 1) Diagnostic: actor loss, critic loss, total MC loss
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, key, color, label in [
        (axes[0], "L_actor",  "steelblue",  "Actor loss"),
        (axes[1], "L_critic", "tomato",     "Critic loss"),
        (axes[2], "L_total",  "darkviolet", "Total MC loss"),
    ]:
        y = smooth(np.array(curve[key]))
        ax.plot(ep_g[SW - 1:], y, color=color, lw=1.5)
        ax.axhline(0, color="black", ls=":", lw=0.7)
        add_phase_labels(ax, len(curve["success"]))
        ax.set_ylabel(label, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].set_xlabel("Episode (global)")
    fig.suptitle(f"A2C diagnostic losses (seed {seed})", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "diagnostic_losses.png"), dpi=150)
    plt.close()

    # 2) Advantage std — should shrink as critic improves
    fig, ax = plt.subplots(figsize=(12, 3))
    y = smooth(np.array(curve["adv_std"]))
    ax.plot(ep_g[SW - 1:], y, color="darkorange", lw=1.5)
    add_phase_labels(ax, len(curve["success"]))
    ax.set_ylabel("Advantage std")
    ax.set_xlabel("Episode (global)")
    ax.set_title(f"Advantage std — should decrease as critic learns (seed {seed})")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "advantage_std.png"), dpi=150)
    plt.close()

    # 3) Success curve
    fig, ax = plt.subplots(figsize=(12, 4))
    y = smooth(np.array(curve["success"]))
    ax.plot(ep_g[SW - 1:], y, lw=1.7)
    add_phase_labels(ax, len(curve["success"]))
    ax.set_ylim(0, 1.1); ax.set_ylabel("Success"); ax.set_xlabel("Episode (global)")
    ax.set_title(f"A2C success curve (seed {seed})")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "success_curve.png"), dpi=150)
    plt.close()

    # 4) Return curve
    fig, ax = plt.subplots(figsize=(12, 4))
    y = smooth(np.array(curve["return"]))
    ax.plot(ep_g[SW - 1:], y, lw=1.7, color="seagreen")
    add_phase_labels(ax, len(curve["success"]))
    ax.set_ylabel("Return"); ax.set_xlabel("Episode (global)")
    ax.set_title(f"A2C return curve (seed {seed})")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "return_curve.png"), dpi=150)
    plt.close()

    # 5) A1 vs A2 success — savings
    phases = split_phases(curve["success"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for ph_idx, col, lbl in [(A1_IDX, "steelblue", "A1"), (A2_IDX, "darkorange", "A2")]:
        y = smooth(np.array(phases[ph_idx]))
        ax.plot(ep_local[SW - 1:], y, lw=2, color=col, label=lbl)
    ax.set_xlim(1, N_PHASE_EP); ax.set_ylim(0, 1.1)
    ax.set_xlabel("Episode within phase"); ax.set_ylabel("Success")
    ax.set_title(f"A1 vs A2 success — savings (seed {seed})")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "a1_vs_a2_success.png"), dpi=150)
    plt.close()

    # 6) A1 vs A2 angle
    angle_phases = split_phases(curve["angle"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for ph_idx, col, lbl in [(A1_IDX, "steelblue", "A1"), (A2_IDX, "darkorange", "A2")]:
        y = smooth(np.array(angle_phases[ph_idx]))
        ep_sm = ep_local[SW - 1:]
        ax.plot(ep_sm[ep_sm <= 400], y[ep_sm <= 400], color=col, lw=2, label=lbl)
    ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° target")
    ax.set_xlim(1, 400); ax.set_ylim(-110, 60)
    ax.set_xlabel("Episode within phase"); ax.set_ylabel("Mean angle (°)")
    ax.set_title(f"A1 vs A2 angle (seed {seed})")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "a1_vs_a2_angle.png"), dpi=150)
    plt.close()

    print(f"Seed {seed}: figures saved to {fig_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    os.makedirs(FIG_DIR, exist_ok=True)

    for seed in range(1):
        print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
        seed_dir   = os.path.join(FIG_DIR, f"seed_{seed}")
        curve_path = os.path.join(seed_dir, "curve.npz")
        os.makedirs(seed_dir, exist_ok=True)

        if os.path.exists(curve_path):
            print("Cached run found, loading.")
            curve = np.load(curve_path, allow_pickle=True)["curve"].item()
        else:
            curve = run_babab(seed=seed)
            np.savez(curve_path, curve=curve)

        plot_seed(seed, curve, seed_dir)

    print(f"\nDone. Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()

    # testing
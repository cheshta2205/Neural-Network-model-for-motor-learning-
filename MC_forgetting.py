"""
MC_forgetting.py

Catastrophic forgetting test.
Compares original MC vs FiLM-gated MC.

Training phases: pre (B) → A → B → A2
During phase B: after each train+eval step on B, also run a no-grad eval on A.
This gives a forgetting curve — how much does A accuracy decay while training on B?

If FiLM is working: A accuracy during B training stays high (protected).
If no FiLM: A accuracy drops (overwritten).
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
MC_INPUT    = 2 + LATENT_DIM + CB_PROJ_DIM + 2

MAX_STEPS   = 20
N_PHASE_EP  = 2000
GOAL_RADIUS = 0.1
GRID_SIZE   = 4.0
DT          = 1.0
LR_MC       = 1e-3

CKPT    = "checkpoints/ectocb_8ctx_seq80_z32.pt"
FIG_DIR = "../../figs/ECtoCB/MC_policy/MC_forgetting"

CTX_A  = (1.0, np.pi/3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B  = (1.0, 0.0,     0.0, 0.0, 0.0, 0.0, "B: rot0")
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
    """Original MC — no FiLM."""
    def __init__(self):
        super().__init__()
        self.hidden_dim = MC_HIDDEN
        self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.cell       = nn.GRUCell(MC_INPUT, MC_HIDDEN)
        self.readout    = nn.Linear(MC_HIDDEN, 2)
        W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
        self.register_buffer("cb_proj", W / W.norm(dim=0, keepdim=True))

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, h_ec, g, h_mc, cb_hidden):
        z_ec = torch.tanh(self.ec_readout(h_ec))
        z_cb = cb_hidden @ self.cb_proj
        inp  = torch.cat([s_t, z_ec, z_cb, g], dim=-1)
        h_new = self.cell(inp, h_mc)
        return torch.tanh(self.readout(h_new)), h_new


class MotorCortexFiLM(nn.Module):
    """MC with FiLM gating — EC modulates scale+shift of every MC neuron."""
    def __init__(self):
        super().__init__()
        self.hidden_dim = MC_HIDDEN
        self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.cell       = nn.GRUCell(MC_INPUT, MC_HIDDEN)
        self.film       = nn.Linear(LATENT_DIM, MC_HIDDEN * 2)
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
        scale, shift = self.film(z_ec).chunk(2, dim=-1)
        h_modulated  = scale * h_new + shift
        return torch.tanh(self.readout(h_modulated)), h_new


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


# ── Single episode ────────────────────────────────────────────────────────────
def run_episode(ec, cb, mc, device, ctx, rng, train=True):
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

        with torch.no_grad() if not train else torch.enable_grad():
            cb_h = cb.hidden(pos.unsqueeze(0), prev_a, z_t)

        a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec, g_t.unsqueeze(0), h_mc, cb_h)
        s_next, vel_next = env_step_torch(
            pos, vel, a_t.squeeze(0), beta, theta, gamma, g, bx, by, device)

        if train:
            step_losses.append(torch.sum((g_t - s_next) ** 2))

        with torch.no_grad():
            if t == 0:
                a_np  = a_t.squeeze(0).detach().cpu().numpy()
                gd    = (g_t - pos).cpu().numpy()
                cross = gd[0] * a_np[1] - gd[1] * a_np[0]
                dot   = a_np[0] * gd[0] + a_np[1] * gd[1]
                first_angle = float(np.degrees(np.arctan2(cross, dot)))

            s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
            eps   = s_next.detach().unsqueeze(0) - cb(pos.unsqueeze(0), a_t.detach(), z_t)
            z_t, h_ec = ec.step(
                torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

        prev_a = a_t.detach()
        pos    = s_next.detach()
        vel    = vel_next
        steps += 1

    loss = torch.stack(step_losses).sum() if step_losses else \
           torch.tensor(0.0, device=device, requires_grad=True)
    first_angle = first_angle if first_angle is not None else 0.0
    return loss, float(success), steps, first_angle


# ── Training loop with cross-context forgetting eval ─────────────────────────
def run_forgetting(ec, cb, mc, device, seed=0):
    """
    Runs pre → A → B → A2 phases.
    During phase B: after each episode, also evals on context A (no training).
    Returns curve dict with 'forget_ep' and 'forget_acc' (A accuracy during B).
    """
    ec.eval(); cb.eval(); mc.train()
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.Adam(mc.parameters(), lr=LR_MC)
    rng       = np.random.default_rng(seed)

    curve = {
        "ep": [], "success": [], "phase": [],
        "forget_ep": [], "forget_acc": [], "forget_angle": [],   # A probe during B
        "a2_ep": [], "a2_acc": [],
    }
    ep_global = 0

    for phase_idx, ctx in enumerate(PHASES):
        label = PHASE_LABELS[phase_idx]
        print(f"\nPhase {label} ({N_PHASE_EP} episodes)...")

        for ep in range(1, N_PHASE_EP + 1):
            ep_global += 1
            mc.train()

            # train on current context
            optimizer.zero_grad()
            loss, _, _, _ = run_episode(ec, cb, mc, device, ctx, rng, train=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            optimizer.step()

            # eval on current context
            mc.eval()
            with torch.no_grad():
                _, succ, _, _ = run_episode(ec, cb, mc, device, ctx, rng, train=False)
            curve["ep"].append(ep_global)
            curve["success"].append(float(succ))
            curve["phase"].append(phase_idx)

            # during B: also eval on A (forgetting probe — acc + angle)
            if phase_idx == 2:
                with torch.no_grad():
                    _, succ_a, _, angle_a = run_episode(ec, cb, mc, device, CTX_A, rng, train=False)
                curve["forget_ep"].append(ep)
                curve["forget_acc"].append(float(succ_a))
                curve["forget_angle"].append(float(angle_a))

            # during A2: track A2 performance
            if phase_idx == 3:
                curve["a2_ep"].append(ep)
                curve["a2_acc"].append(float(succ))

            if ep % 500 == 0:
                recent = np.mean(curve["success"][-100:])
                print(f"  [{label}] ep {ep:4d} | acc (last 100)={recent:.2f}")

    return curve


# ── Plots ─────────────────────────────────────────────────────────────────────
def smooth(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_forgetting(curve_orig, curve_film):
    os.makedirs(FIG_DIR, exist_ok=True)
    SW = 20
    ep = np.arange(1, N_PHASE_EP + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # ── Left: A accuracy during B training ────────────────────────────────────
    ax = axes[0]
    for curve, color, label in [
        (curve_orig, "steelblue", "Original MC"),
        (curve_film, "tomato",    "FiLM MC"),
    ]:
        y    = np.array(curve["forget_acc"])
        y_s  = smooth(y, SW)
        ep_s = ep[SW-1:]
        ax.plot(ep_s, y_s, color=color, lw=2, label=label)

    ax.set_xlabel("Episode within B phase")
    ax.set_ylabel("Success rate on context A")
    ax.set_title("Forgetting: A accuracy while training on B\n(higher = less forgetting)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Middle: A first-step angle during B training ───────────────────────────
    ax = axes[1]
    for curve, color, label in [
        (curve_orig, "steelblue", "Original MC"),
        (curve_film, "tomato",    "FiLM MC"),
    ]:
        y    = np.array(curve["forget_angle"])
        y_s  = smooth(y, SW)
        ep_s = ep[SW-1:]
        ax.plot(ep_s, y_s, color=color, lw=2, label=label)

    ax.axhline(-60, color="gray", ls="--", lw=1.2, label="−60° (A target)")
    ax.set_xlabel("Episode within B phase")
    ax.set_ylabel("First-step angle on context A (deg)")
    ax.set_title("Forgetting: A action angle while training on B\n(closer to −60° = less forgetting)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Right: summary — end-of-A → end-of-B probe → end-of-A2 ───────────────
    ax = axes[2]
    for curve, color, label in [
        (curve_orig, "steelblue", "Original MC"),
        (curve_film, "tomato",    "FiLM MC"),
    ]:
        phase_a_succ = np.array(curve["success"])[N_PHASE_EP:2*N_PHASE_EP]
        acc_end_a    = np.mean(phase_a_succ[-100:])
        acc_during_b = np.mean(np.array(curve["forget_acc"])[-100:])
        acc_end_a2   = np.mean(np.array(curve["success"])[3*N_PHASE_EP:][-100:])

        xs = [1, 2, 3]
        ys = [acc_end_a, acc_during_b, acc_end_a2]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=8, label=label)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["End of A", "End of B\n(A probe)", "End of A2"])
    ax.set_ylabel("Success rate on context A")
    ax.set_title("Context A accuracy across phases\n(drop = forgetting, recovery = savings)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Catastrophic forgetting test: Original MC vs FiLM-gated MC", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/forgetting_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {FIG_DIR}/forgetting_comparison.png")


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
    data_path = f"{FIG_DIR}/forgetting_data.npz"

    if os.path.exists(data_path) and "--replot" in sys.argv:
        print("Loading saved data — skipping training.")
        d          = np.load(data_path, allow_pickle=True)
        curve_orig = d["curve_orig"].item()
        curve_film = d["curve_film"].item()
    else:
        mc_orig = MotorCortex().to(device)
        mc_film = MotorCortexFiLM().to(device)

        print("\n=== ORIGINAL MC ===")
        curve_orig = run_forgetting(ec, cb, mc_orig, device, seed=0)
        print("\n=== FiLM MC ===")
        curve_film = run_forgetting(ec, cb, mc_film, device, seed=0)

        np.savez(data_path, curve_orig=curve_orig, curve_film=curve_film)
        print(f"\nSaved data to {FIG_DIR}/")

    plot_forgetting(curve_orig, curve_film)
    print("Done.")


if __name__ == "__main__":
    main()

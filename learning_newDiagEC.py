"""
learning_newDiagEC.py

EC+CB retrained on 8 contexts (axis-aligned + diagonal ±0.4).
MC trained on familiar 4 axis-aligned ±0.4 only (full s_t, no delay).
Evaluated on familiar (axis-aligned) vs novel (diagonal ±0.4).

Key idea for dissociation:
  - MC's h_mc was trained on axis-aligned force fields → knows how to handle them.
    With EC lesion on familiar: h_mc self-infers context from familiar trajectory
    patterns → still succeeds.
  - MC's h_mc has NEVER seen diagonal force fields → cannot self-infer diagonal
    context from trajectory alone. Without EC (z_t = 0), it is blind to which
    diagonal field it's in → fails on novel.
  - EC knows all 8 contexts (newly retrained) → z_t is informative for diagonal →
    MC with intact EC succeeds on novel too.

Desired result:
  familiar | EC intact  | high success
  familiar | EC lesion  | high success   (MC h_mc knows these)
  novel    | EC intact  | high success   (EC's z_t guides MC)
  novel    | EC lesion  | low success    (MC blind to diagonal context)

MC input: [s_t(2), z_t'(16), g(2)] = 20 dims
Checkpoint: checkpoints/ectocb_mse_shiftedv2_wDiag.pt
Figures saved to: figs/ECtoCB/gridworld_policy/learning_newDiagEC/
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Hyperparameters ────────────────────────────────────────────────────────────
LATENT_DIM  = 16
EC_HIDDEN   = 256
CB_HIDDEN   = 256
EC_INPUT    = 8
MC_HIDDEN   = 256
MC_INPUT    = 20   # s_t(2) + z_t'(16) + g(2)

MAX_STEPS   = 50
N_TRAIN_EP  = 10000
N_EVAL_EP   = 500
GOAL_RADIUS = 0.5
GRID_SIZE   = 4
LR_MC       = 1e-3
DIST_LAMBDA = 0.1

FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/gridworld_policy/learning_newDiagEC"
CKPT    = "checkpoints/ectocb_mse_shiftedv2_wDiag.pt"

ACTIONS   = [(-1., 0.), (1., 0.), (0., -1.), (0., 1.)]
N_ACTIONS = len(ACTIONS)

# MC trains on familiar 4 axis-aligned contexts only
CONTEXTS = [
    ( 0.4,  0.0),
    (-0.4,  0.0),
    ( 0.0,  0.4),
    ( 0.0, -0.4),
]

# Novel: diagonal ±0.4 — EC knows these, MC's h_mc has never seen them
NOVEL_CONTEXTS = [
    ( 0.4,  0.4),
    (-0.4,  0.4),
    ( 0.4, -0.4),
    (-0.4, -0.4),
]


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
        self.net = nn.Sequential(
            nn.Linear(4 + LATENT_DIM, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, CB_HIDDEN),       nn.ReLU(),
            nn.Linear(CB_HIDDEN, 2),
        )

    def forward(self, s, a, z):
        return self.net(torch.cat([s, a, z], dim=-1))


class MotorCortex(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = MC_HIDDEN
        self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.cell       = nn.GRUCell(MC_INPUT, MC_HIDDEN)
        self.readout    = nn.Linear(MC_HIDDEN, N_ACTIONS)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s, h_ec, g, h):
        # s = current s_t (full, no delay)
        z_prime = torch.tanh(self.ec_readout(h_ec))
        h_new   = self.cell(torch.cat([s, z_prime, g], dim=-1), h)
        pi      = torch.softmax(self.readout(h_new), dim=-1)
        return pi, h_new


# ── Environment ────────────────────────────────────────────────────────────────
def env_step(x, y, dx, dy, beta, gamma):
    a      = np.array([dx, dy], dtype=np.float32)
    a_norm = np.linalg.norm(a)
    a_hat  = a / a_norm if a_norm > 0 else np.zeros(2, dtype=np.float32)
    a_perp = np.array([-a_hat[1], a_hat[0]], dtype=np.float32)
    f_env  = beta * a_hat + gamma * a_perp
    return (float(np.clip(x + dx + f_env[0], 0, GRID_SIZE)),
            float(np.clip(y + dy + f_env[1], 0, GRID_SIZE)))


# ── DAgger episode ─────────────────────────────────────────────────────────────
def run_dagger_episode(ec, cb, mc, device, rng):
    start = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    goal  = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    while np.array_equal(start, goal):
        goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    beta, gamma = CONTEXTS[rng.integers(0, len(CONTEXTS))]

    g_tensor = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
    x, y     = float(start[0]), float(start[1])
    h_ec     = ec.init_hidden(1, device)
    h_mc     = mc.init_hidden(1, device)
    z_t      = torch.zeros(1, LATENT_DIM, device=device)
    step_losses = []

    for t in range(MAX_STEPS):
        s_np = np.array([x, y], dtype=np.float32)
        if np.linalg.norm(s_np - goal) < GOAL_RADIUS:
            break

        s_t = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)

        # Teacher: oracle picks optimal action given true beta/gamma
        a_teacher = int(np.argmin([
            np.sum((np.array(env_step(x, y, dx, dy, beta, gamma)) - goal)**2)
            for dx, dy in ACTIONS
        ]))

        # MC forward — gets current s_t + EC's z_t
        pi, h_mc = mc.step(s_t, h_ec, g_tensor, h_mc)
        bc_loss  = -torch.log(pi.squeeze(0)[a_teacher] + 1e-8)

        # Differentiable distance loss: expected next state under pi
        all_next = torch.tensor(
            [list(env_step(x, y, dx, dy, beta, gamma)) for dx, dy in ACTIONS],
            dtype=torch.float32, device=device)
        g_t           = torch.tensor(goal, dtype=torch.float32, device=device)
        expected_next = (pi.squeeze(0).unsqueeze(1) * all_next).sum(0)
        dist_loss     = torch.norm(expected_next - g_t)
        step_losses.append(bc_loss + DIST_LAMBDA * dist_loss)

        # DAgger: MC's action drives the agent
        mc_action      = torch.argmax(pi.squeeze(0)).item()
        dx_mc, dy_mc   = ACTIONS[mc_action]
        x_next, y_next = env_step(x, y, dx_mc, dy_mc, beta, gamma)

        # EC updates from current transition (real-time, no delay)
        with torch.no_grad():
            s_tm1     = torch.zeros(1, 2, device=device) if t == 0 else torch.tensor([[x, y]], dtype=torch.float32, device=device)
            a_tm1     = torch.zeros(1, 2, device=device) if t == 0 else torch.tensor([[dx_mc, dy_mc]], dtype=torch.float32, device=device)
            a_t       = torch.tensor([[dx_mc, dy_mc]], dtype=torch.float32, device=device)
            s_next    = torch.tensor([[x_next, y_next]], dtype=torch.float32, device=device)
            eps       = s_next - cb(s_t, a_t, z_t)
            z_t, h_ec = ec.step(torch.cat([s_tm1, a_tm1, s_t, eps], dim=-1), h_ec)

        x, y = x_next, y_next

    if not step_losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(step_losses).sum()


# ── Train MC ───────────────────────────────────────────────────────────────────
def train_mc(ec, cb, mc, device, seed=0):
    ec.eval(); cb.eval(); mc.train()
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.Adam(mc.parameters(), lr=LR_MC)
    rng       = np.random.default_rng(seed)
    losses    = []

    print(f"\nTraining MC — DAgger, familiar axis-aligned only ({N_TRAIN_EP} episodes)...")
    for ep in range(1, N_TRAIN_EP + 1):
        optimizer.zero_grad()
        loss = run_dagger_episode(ec, cb, mc, device, rng)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if ep % 300 == 0:
            print(f"  Ep {ep:4d} | loss={np.mean(losses[-300:]):.2f}")

    return losses


# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(ec, cb, mc, device, contexts, seed=42, ec_lesion=False):
    ec.eval(); cb.eval(); mc.eval()
    rng     = np.random.default_rng(seed)
    results = {"success": [], "steps": []}

    for _ in range(N_EVAL_EP):
        start = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
        goal  = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
        while np.array_equal(start, goal):
            goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
        beta, gamma = contexts[rng.integers(0, len(contexts))]

        g_tensor = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
        x, y     = float(start[0]), float(start[1])
        h_ec     = ec.init_hidden(1, device)
        h_mc     = mc.init_hidden(1, device)
        z_t      = torch.zeros(1, LATENT_DIM, device=device)
        success  = False
        steps    = 0

        with torch.no_grad():
            for t in range(MAX_STEPS):
                if np.linalg.norm(np.array([x, y]) - goal) < GOAL_RADIUS:
                    success = True; break

                s_t = torch.tensor([[x, y]], dtype=torch.float32, device=device)
                pi, h_mc   = mc.step(s_t, h_ec, g_tensor, h_mc)
                action_idx = torch.argmax(pi.squeeze(0)).item()
                dx, dy     = ACTIONS[action_idx]
                x_next, y_next = env_step(x, y, dx, dy, beta, gamma)

                if not ec_lesion:
                    s_tm1     = torch.zeros(1, 2, device=device) if t == 0 else torch.tensor([[x, y]], dtype=torch.float32, device=device)
                    a_tm1     = torch.zeros(1, 2, device=device) if t == 0 else torch.tensor([[dx, dy]], dtype=torch.float32, device=device)
                    a_t       = torch.tensor([[dx, dy]], dtype=torch.float32, device=device)
                    s_next_t  = torch.tensor([[x_next, y_next]], dtype=torch.float32, device=device)
                    eps       = s_next_t - cb(s_t, a_t, z_t)
                    z_t, h_ec = ec.step(torch.cat([s_tm1, a_tm1, s_t, eps], dim=-1), h_ec)

                x, y   = x_next, y_next
                steps += 1

        results["success"].append(float(success))
        results["steps"].append(steps)

    return results


# ── Evaluate by timestep ───────────────────────────────────────────────────────
def evaluate_by_timestep(ec, cb, mc, device, contexts, seed=99, ec_lesion=False):
    ec.eval(); cb.eval(); mc.eval()
    rng         = np.random.default_rng(seed)
    dist_matrix = np.full((N_EVAL_EP, MAX_STEPS), np.nan)

    with torch.no_grad():
        for ep in range(N_EVAL_EP):
            start = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
            goal  = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
            while np.array_equal(start, goal):
                goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
            beta, gamma = contexts[rng.integers(0, len(contexts))]

            g_tensor = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
            x, y     = float(start[0]), float(start[1])
            h_ec     = ec.init_hidden(1, device)
            h_mc     = mc.init_hidden(1, device)
            z_t      = torch.zeros(1, LATENT_DIM, device=device)

            for t in range(MAX_STEPS):
                dist = np.linalg.norm(np.array([x, y]) - goal)
                if dist < GOAL_RADIUS:
                    break
                dist_matrix[ep, t] = dist

                s_t = torch.tensor([[x, y]], dtype=torch.float32, device=device)
                pi, h_mc   = mc.step(s_t, h_ec, g_tensor, h_mc)
                action_idx = torch.argmax(pi.squeeze(0)).item()
                dx, dy     = ACTIONS[action_idx]
                x_next, y_next = env_step(x, y, dx, dy, beta, gamma)

                if not ec_lesion:
                    s_tm1     = torch.zeros(1, 2, device=device) if t == 0 else torch.tensor([[x, y]], dtype=torch.float32, device=device)
                    a_tm1     = torch.zeros(1, 2, device=device) if t == 0 else torch.tensor([[dx, dy]], dtype=torch.float32, device=device)
                    a_t       = torch.tensor([[dx, dy]], dtype=torch.float32, device=device)
                    s_next_t  = torch.tensor([[x_next, y_next]], dtype=torch.float32, device=device)
                    eps       = s_next_t - cb(s_t, a_t, z_t)
                    z_t, h_ec = ec.step(torch.cat([s_tm1, a_tm1, s_t, eps], dim=-1), h_ec)

                x, y = x_next, y_next

    return dist_matrix


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_training(losses):
    os.makedirs(FIG_DIR, exist_ok=True)
    smoothed = np.convolve(losses, np.ones(50)/50, mode='valid')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(smoothed, color="steelblue", lw=2)
    ax.set_xlabel("Episode"); ax.set_ylabel("Loss")
    ax.set_title("MC training loss — DAgger, familiar axis-aligned only\n"
                 "EC+CB trained on 8 contexts (axis-aligned + diagonal)")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "mc_training_loss.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


def plot_2x2(results):
    os.makedirs(FIG_DIR, exist_ok=True)
    labels_x   = ["Familiar ±0.4\n(axis-aligned)", "Novel ±0.4\n(diagonal)"]
    bar_labels = ["EC intact", "EC lesion"]
    colors     = ["steelblue", "tomato"]
    x          = np.arange(len(labels_x))
    w          = 0.35

    success = {
        "EC intact": [np.mean(results["familiar_ec"]["success"]),
                      np.mean(results["novel_ec"]["success"])],
        "EC lesion": [np.mean(results["familiar_lesion"]["success"]),
                      np.mean(results["novel_lesion"]["success"])],
    }
    steps = {
        "EC intact": [np.mean(results["familiar_ec"]["steps"]),
                      np.mean(results["novel_ec"]["steps"])],
        "EC lesion": [np.mean(results["familiar_lesion"]["steps"]),
                      np.mean(results["novel_lesion"]["steps"])],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, ylabel in zip(axes, [success, steps], ["Success rate", "Mean steps to goal"]):
        for i, (lbl, color) in enumerate(zip(bar_labels, colors)):
            vals = metric[lbl]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=lbl, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                fmt = f"{val:.2f}" if ylabel == "Success rate" else f"{val:.1f}"
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.02 if ylabel == "Success rate" else 0.5),
                        fmt, ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels_x)
        ax.set_ylabel(ylabel)
        if ylabel == "Success rate":
            ax.set_ylim(0, 1.15)
        ax.legend()

    fig.suptitle("EC lesion — familiar axis-aligned vs novel diagonal\n"
                 "MC trained on familiar only | EC+CB trained on all 8 contexts\n"
                 "EC lesion: z_t = 0 throughout episode",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "2x2_newDiagEC.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


def plot_timestep(dist_intact, dist_lesion, label="novel"):
    os.makedirs(FIG_DIR, exist_ok=True)
    mean_intact = np.nanmean(dist_intact, axis=0)
    mean_lesion = np.nanmean(dist_lesion, axis=0)
    sem_intact  = np.nanstd(dist_intact, axis=0) / np.sqrt(np.sum(~np.isnan(dist_intact), axis=0).clip(1))
    sem_lesion  = np.nanstd(dist_lesion, axis=0) / np.sqrt(np.sum(~np.isnan(dist_lesion), axis=0).clip(1))

    n_valid = np.maximum(np.sum(~np.isnan(dist_intact), axis=0),
                         np.sum(~np.isnan(dist_lesion), axis=0))
    cutoff  = min(int(np.searchsorted(-n_valid, -(N_EVAL_EP * 0.1))), MAX_STEPS)
    steps   = np.arange(cutoff)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, mean_intact[:cutoff], color="steelblue", lw=2, label="EC intact")
    ax.fill_between(steps, mean_intact[:cutoff] - sem_intact[:cutoff],
                    mean_intact[:cutoff] + sem_intact[:cutoff], color="steelblue", alpha=0.2)
    ax.plot(steps, mean_lesion[:cutoff], color="tomato", lw=2, label="EC lesion")
    ax.fill_between(steps, mean_lesion[:cutoff] - sem_lesion[:cutoff],
                    mean_lesion[:cutoff] + sem_lesion[:cutoff], color="tomato", alpha=0.2)
    ax.set_xlabel("Timestep within episode")
    ax.set_ylabel("Distance to goal")
    ax.set_title(f"Within-episode adaptation — {label}\n"
                 "MC trained on familiar axis-aligned | EC+CB trained on 8 contexts")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"timestep_{label}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(CKPT, map_location=device)
    ec   = EntorhinalCortex().to(device)
    cb   = CerebellumCB().to(device)
    ec.load_state_dict(ckpt["ec"])
    cb.load_state_dict(ckpt["cb"])
    ec.eval(); cb.eval()
    print(f"Loaded frozen EC + CB from {CKPT}")
    print(f"MC input: [s_t(2), z_t'(16), g(2)] — full current state, no delay")

    mc = MotorCortex().to(device)
    print(f"MC parameters: {sum(p.numel() for p in mc.parameters()):,}")
    losses = train_mc(ec, cb, mc, device)
    plot_training(losses)

    print("\nEvaluating 4 conditions...")
    results = {}
    conds = [
        ("familiar_ec",     CONTEXTS,       False),
        ("familiar_lesion", CONTEXTS,       True),
        ("novel_ec",        NOVEL_CONTEXTS, False),
        ("novel_lesion",    NOVEL_CONTEXTS, True),
    ]
    for key, ctx, lesion in conds:
        r = evaluate(ec, cb, mc, device, ctx, ec_lesion=lesion)
        results[key] = r
        tag = "EC intact" if not lesion else "EC lesion"
        ff  = "familiar" if ctx is CONTEXTS else "novel   "
        print(f"  {ff} | {tag:10s} | "
              f"success={np.mean(r['success']):.3f}  steps={np.mean(r['steps']):.1f}")

    plot_2x2(results)

    print("\nWithin-episode adaptation...")
    d_fam_intact = evaluate_by_timestep(ec, cb, mc, device, CONTEXTS,       ec_lesion=False)
    d_fam_lesion = evaluate_by_timestep(ec, cb, mc, device, CONTEXTS,       ec_lesion=True)
    plot_timestep(d_fam_intact, d_fam_lesion, label="familiar")

    d_nov_intact = evaluate_by_timestep(ec, cb, mc, device, NOVEL_CONTEXTS, ec_lesion=False)
    d_nov_lesion = evaluate_by_timestep(ec, cb, mc, device, NOVEL_CONTEXTS, ec_lesion=True)
    plot_timestep(d_nov_intact, d_nov_lesion, label="novel_diagonal")

    print("\nDone.")


if __name__ == "__main__":
    main()

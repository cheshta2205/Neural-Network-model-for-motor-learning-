
"""
diag_hc_weights.py

Test H2: does B training overwrite A in the Hebbian weight matrix?

Runs the HC condition (MC+EC+CB+HC) for seeds 0 (good) and 1 (bad),
saves hc.A.weight after each phase (pre-B, A, B), then measures:

  response_A = mean ||tanh(W @ key_A)|| over A-context z_t keys
  response_B = mean ||tanh(W @ key_B)|| over B-context z_t keys

If after B phase: response_B >> response_A  → H2 confirmed (B dominates)
If after B phase: response_A still strong   → H2 not the issue
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── copy hyperparams from mc_hebbian_zHC.py ──────────────────────────────────
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
HC_ALPHA     = 0.1
KEY_DIM      = EC_HIDDEN + 2

CKPT    = "checkpoints/ectocb_8ctx_seq80_z32.pt"
FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/MC_EC_CB/MC_savings_norouting/hebbian/diag_weights"

CTX_A = (1.0, np.pi / 3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B = (1.0, 0.0,      +0.4, 0.0, 0.0, 0.0, "B: curl+")
PHASES       = [CTX_B, CTX_A, CTX_B, CTX_A, CTX_B]
PHASE_LABELS = ["pre-B", "A", "B", "A2", "B2"]
A_IDX = 1
B_IDX = 2


# ── Model definitions (matching mc_hebbian_zHC.py) ────────────────────────────
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
        return self.out(self.layer2(self.layer1(torch.cat([s, a], dim=-1))))
    def hidden(self, s, a):
        return self.layer2(self.layer1(torch.cat([s, a], dim=-1)))

class MotorCortex(nn.Module):
    def __init__(self, use_ec=True, use_cb=True):
        super().__init__()
        self.use_ec = use_ec; self.use_cb = use_cb
        input_dim = 2 + 2
        if use_ec:
            input_dim += LATENT_DIM
            self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        if use_cb:
            input_dim += CB_PROJ_DIM
            W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
            self.register_buffer("cb_proj", W / W.norm(dim=0, keepdim=True))
        self.cell    = nn.GRUCell(input_dim, MC_HIDDEN)
        self.readout = nn.Linear(MC_HIDDEN, 2)
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)
    def step(self, s_t, h_ec, g, h_mc, cb_hidden=None):
        parts = [s_t]
        if self.use_ec: parts.append(torch.tanh(self.ec_readout(h_ec)))
        if self.use_cb and cb_hidden is not None: parts.append(cb_hidden @ self.cb_proj)
        parts.append(g)
        h_new = self.cell(torch.cat(parts, dim=-1), h_mc)
        return torch.tanh(self.readout(h_new)), h_new

class HippocampusRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Linear(KEY_DIM, HC_HIDDEN, bias=False)
        self.B = nn.Linear(HC_HIDDEN, 2,       bias=False)
        nn.init.zeros_(self.A.weight)
        nn.init.zeros_(self.B.weight)
        for p in self.parameters(): p.requires_grad_(False)
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, HC_HIDDEN, device=device)
    def step(self, h_ec, g, h_hc):
        key  = torch.cat([h_ec, g], dim=-1)
        h_hc = (1 - HC_ALPHA) * h_hc + HC_ALPHA * torch.tanh(self.A(key))
        a_hc = torch.tanh(self.B(h_hc))
        return a_hc, h_hc, key


def env_step(pos, vel, action, beta, theta, gamma, g, bx, by, device):
    cos_t = torch.tensor(float(np.cos(theta)), dtype=torch.float32, device=device)
    sin_t = torch.tensor(float(np.sin(theta)), dtype=torch.float32, device=device)
    R    = torch.stack([torch.stack([cos_t, -sin_t]), torch.stack([sin_t, cos_t])])
    curl = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32, device=device)
    b_v  = torch.tensor([float(bx), float(by)], dtype=torch.float32, device=device)
    F    = beta * (R @ action) + gamma * (curl @ vel) + g * b_v
    vel_new = vel + F
    return torch.clamp(pos + DT * vel_new, 0, GRID_SIZE), vel_new.detach()


def hc_hebbian_update(hc, hc_buffer, lr):
    with torch.no_grad():
        for key_t, h_t, a_mc_t in hc_buffer:
            hc.A.weight.data += lr * torch.outer(h_t.squeeze(0), key_t.squeeze(0))
            hc.B.weight.data += lr * torch.outer(a_mc_t, h_t.squeeze(0))


def run_episode(ec, cb, mc, hc, device, ctx, rng, train=True):
    beta, theta, gamma, g, bx, by, _ = ctx
    start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(start - goal) < 0.5:
        goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    pos  = torch.tensor(start, dtype=torch.float32, device=device)
    vel  = torch.zeros(2, dtype=torch.float32, device=device)
    g_t  = torch.tensor(goal,  dtype=torch.float32, device=device)
    h_ec = ec.init_hidden(1, device)
    h_mc = mc.init_hidden(1, device)
    h_hc = hc.init_hidden(1, device)
    z_t  = torch.zeros(1, LATENT_DIM, device=device)
    prev_a = torch.zeros(1, 2, device=device)

    goal_losses = []; hc_losses = []; hc_buffer = []
    success = False; steps = 0

    for t in range(MAX_STEPS):
        if torch.norm(pos - g_t).item() < GOAL_RADIUS:
            success = True; break

        cb_h = cb.hidden(pos.unsqueeze(0), prev_a, z_t)
        a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec, g_t.unsqueeze(0), h_mc, cb_h)
        s_next, vel_next = env_step(pos, vel, a_t.squeeze(0), beta, theta, gamma, g, bx, by, device)

        if train:
            goal_losses.append(torch.sum((g_t - s_next) ** 2))
            with torch.no_grad():
                a_hc, h_hc, key_t = hc.step(h_ec, g_t.unsqueeze(0), h_hc)
            hc_losses.append(HC_KL_WEIGHT * torch.sum((a_t.squeeze(0) - a_hc.squeeze(0).detach()) ** 2))
            hc_buffer.append((key_t.detach().clone(), h_hc.detach().clone(), a_t.squeeze(0).detach().clone()))

        with torch.no_grad():
            s_tm1   = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
            cb_pred = cb(pos.unsqueeze(0), a_t.detach(), z_t)
            eps     = s_next.detach().unsqueeze(0) - cb_pred
            z_t, h_ec = ec.step(torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

        prev_a = a_t.detach(); pos = s_next.detach(); vel = vel_next; steps += 1

    loss = torch.stack(goal_losses + hc_losses).sum() if (goal_losses or hc_losses) else torch.tensor(0.0, device=device, requires_grad=True)
    return loss, success, hc_buffer


# ── Collect z_t keys under random actions ─────────────────────────────────────
def collect_keys(ec, cb, ctx, n_episodes, device, rng):
    beta, theta, gamma, g_coef, bx, by, _ = ctx
    keys = []
    with torch.no_grad():
        for _ in range(n_episodes):
            start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
            goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
            while np.linalg.norm(start - goal) < 0.5:
                goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
            pos  = torch.tensor(start, dtype=torch.float32, device=device)
            vel  = torch.zeros(2, dtype=torch.float32, device=device)
            g_t  = torch.tensor(goal,  dtype=torch.float32, device=device)
            h_ec = ec.init_hidden(1, device)
            z_t  = torch.zeros(1, LATENT_DIM, device=device)
            prev_a = torch.zeros(1, 2, device=device)
            for t in range(MAX_STEPS):
                action = torch.tensor(rng.uniform(-1, 1, size=2).astype(np.float32), device=device)
                s_next, vel_new = env_step(pos, vel, action, beta, theta, gamma, g_coef, bx, by, device)
                s_tm1   = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                cb_pred = cb(pos.unsqueeze(0), action.unsqueeze(0), z_t)
                eps     = s_next.unsqueeze(0) - cb_pred
                z_t, h_ec = ec.step(torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)
                if t >= 5:  # only collect context-informative timesteps
                    key = torch.cat([h_ec, g_t.unsqueeze(0)], dim=-1)
                    keys.append(key.squeeze(0).cpu().numpy())
                prev_a = action.unsqueeze(0); pos = s_next.detach(); vel = vel_new.detach()
    return np.array(keys)


# ── Measure HC response to A-keys vs B-keys given weight matrix W ─────────────
def measure_response(W_tensor, keys):
    """mean ||tanh(W @ key)|| over all keys"""
    with torch.no_grad():
        keys_t = torch.tensor(keys, dtype=torch.float32)
        h = torch.tanh(keys_t @ W_tensor.T)  # (N, HC_HIDDEN)
        return h.norm(dim=-1).mean().item()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    device = torch.device("cpu")

    ckpt = torch.load(CKPT, map_location=device)
    ec = EntorhinalCortex().to(device)
    cb = CerebellumCB().to(device)
    ec.load_state_dict(ckpt["ec"])
    cb_sd  = ckpt["cb"]
    remap  = {"net.0": "layer1.0", "net.2": "layer2.0", "net.4": "out"}
    new_sd = {}
    for k, v in cb_sd.items():
        prefix = ".".join(k.split(".")[:2])
        new_sd[k.replace(prefix, remap[prefix]) if prefix in remap else k] = v
    cb.load_state_dict(new_sd)
    ec.eval(); cb.eval()

    # collect A and B keys once (for probing weight matrix)
    rng_keys = np.random.default_rng(999)
    print("Collecting probe keys...")
    keys_A = collect_keys(ec, cb, CTX_A, 100, device, rng_keys)
    keys_B = collect_keys(ec, cb, CTX_B, 100, device, rng_keys)
    print(f"  keys_A: {keys_A.shape}, keys_B: {keys_B.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("H2 test: does B training overwrite A in HC weight matrix?", fontsize=12)

    for ax, seed in zip(axes, [0, 1]):
        label = "good seed" if seed == 0 else "bad seed"
        print(f"\n--- Seed {seed} ({label}) ---")

        mc  = MotorCortex(use_ec=True, use_cb=True).to(device)
        hc  = HippocampusRNN().to(device)
        cb_noec = None

        param_groups = [{"params": mc.parameters(), "lr": LR_MC}]
        optimizer = torch.optim.Adam(param_groups)
        rng = np.random.default_rng(seed)

        W_snapshots = {}   # phase_label → W matrix
        resp_A_history = []
        resp_B_history = []
        phase_ticks = []

        for phase_idx, ctx in enumerate(PHASES):
            label_p = PHASE_LABELS[phase_idx]
            print(f"  Phase {label_p}...")

            for ep in range(1, N_PHASE_EP + 1):
                mc.train(); optimizer.zero_grad()
                loss, succ, hc_buffer = run_episode(ec, cb, mc, hc, device, ctx, rng, train=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
                optimizer.step()
                if succ and hc_buffer:
                    hc_hebbian_update(hc, hc_buffer, LR_HC)

            # snapshot weights after this phase
            W = hc.A.weight.data.clone().cpu()
            W_snapshots[label_p] = W
            r_A = measure_response(W, keys_A)
            r_B = measure_response(W, keys_B)
            resp_A_history.append(r_A)
            resp_B_history.append(r_B)
            phase_ticks.append(label_p)
            print(f"    response_A={r_A:.4f}  response_B={r_B:.4f}  ratio_B/A={r_B/r_A if r_A>0 else 'inf':.2f}")

        xs = np.arange(len(PHASE_LABELS))
        ax.plot(xs, resp_A_history, 'o-', color="steelblue", lw=2, label="response to A-keys")
        ax.plot(xs, resp_B_history, 'o-', color="tomato",    lw=2, label="response to B-keys")
        ax.set_xticks(xs); ax.set_xticklabels(PHASE_LABELS, fontsize=9)
        ax.set_xlabel("After phase", fontsize=9)
        ax.set_ylabel("Mean ||tanh(W @ key)||", fontsize=9)
        ax.set_title(f"Seed {seed} ({'good' if seed==0 else 'bad'})", fontsize=10)
        ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = f"{FIG_DIR}/hc_weight_response.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == "__main__":
    main()

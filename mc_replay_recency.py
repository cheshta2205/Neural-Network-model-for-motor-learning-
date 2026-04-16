"""
mc_replay_hc.py

BABAB motor-learning with A2C (actor-critic) MC and 6 lesion conditions.

Conditions
----------
1) MC only              — no EC, no CB input to MC
2) MC + CB              — CB retrained without EC; MC gets s, g, z_cb
3) MC + EC              — EC uses CB internally; MC gets s, g, z_ec (no z_cb)
4) MC + EC + CB         — full minus HC (A2C baseline)
5) MC + EC + CB + HC    — full brain; HC trained on replay, pulls MC actions
6) MTL lesion           — remove EC & HC; MC gets s, g, z_cb (frozen CB, z_ec=0)

MC loss
-------
    A2C: L_actor + CRITIC_W * L_critic - ENTROPY_W * L_entropy
    + HC_ACT_WEIGHT * ||a_mc - a_hc||²  (condition 5 only)

HC replay
---------
    Buffer stores full episodes.  HC samples REPLAY_K_EP episodes,
    trains GRU to predict stored actions via MSE.
    During live episodes, MC actor is pulled toward HC's action suggestion.
"""

import os
import math
import json
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.distributions import Normal
from scipy.optimize import curve_fit


# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM   = 32
EC_HIDDEN    = 256
CB_HIDDEN    = 256
DG_DIM       = 128
DG_SPARSITY  = 0.10
HC_HIDDEN    = 256
EC_INPUT     = 8
CB_PROJ_DIM  = 32
MC_HIDDEN    = 256

MAX_STEPS    = 40
N_PHASE_EP   = 3500
GOAL_RADIUS  = 1.0
GRID_SIZE    = 4.0
DT           = 1.0

LR_MC            = 3e-4
LR_HC            = 5e-4
LR_CB_NOEC       = 1e-3
GAMMA            = 0.97
ENTROPY_W        = 0.01
ACTION_STD_MIN   = -1.5   # v2 fix: was -2.5; raise floor to keep exploration alive
ACTION_STD_MAX   = -0.2
CRITIC_W         = 1.0
BATCH_EPS        = 16
HC_ACT_WEIGHT    = 0.30
HC_PULL_GATE_N   = 128    # ramp HC pull from 0 → full as same-family pool grows from 0 → 128 episodes

SUCCESS_BONUS    = 5.0

REPLAY_K_EP         = 256       # episodes sampled per HC replay step
REPLAY_MIN_EP       = 20
REPLAY_BUFFER_SIZE  = 7000      # 2 × N_PHASE_EP — holds exactly 2 phases,
                                # so the buffer always contains ~50/50 family balance

# Recency-weighted replay (Gaussian over Δt = current_ep - episode_ep).
#   p(i) ∝ exp(-Δt_i^2 / REPLAY_GAMMA)
# Constant wide γ: σ = sqrt(γ/2) ≈ 15800 ≫ buffer span (8000)
# → effectively uniform across the buffer → ~50/50 family balance always.
# Combined with buffer size = 2 phases, this gives sustained cross-phase
# rehearsal throughout every phase (no fade-out within phase).
REPLAY_GAMMA  = 5.0e8

# CB_noEC training
CB_NOEC_STEPS       = 4000
CB_NOEC_BATCH_SIZE  = 256

CKPT          = "checkpoints/ectocb_8ctx_seq80_z32.pt"
CB_NOEC_CKPT  = "checkpoints/cb_noec_retrained.pt"
FIG_DIR       = "figs/mc_replay_recency_r1_v3"   # v3: r=1.0, constant γ, 5k buffer (50/50), ramp-to-128, 2500 ep/phase

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Contexts / phases ─────────────────────────────────────────────────────────
CTX_A = (1.0, np.pi / 3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B = (1.0, 0.0,      +0.4, 0.0, 0.0, 0.0, "B: curl+")

PHASES       = [CTX_B, CTX_A, CTX_B, CTX_A, CTX_B]
PHASE_LABELS = ["B1: curl+", "A1: rot+60°", "B2: curl+", "A2: rot+60°", "B3: curl+"]

B1_IDX, A1_IDX, B2_IDX, A2_IDX, B3_IDX = 0, 1, 2, 3, 4
A_IDX, B_IDX = A1_IDX, B2_IDX
PHASE_FAMILY = {0: "B", 1: "A", 2: "B", 3: "A", 4: "B"}


# ── Conditions ────────────────────────────────────────────────────────────────
CONDITIONS = [
    {"name": "MC only",           "use_ec": False, "use_cb": False, "use_hc": False, "use_replay": False, "cb_mode": "none"},
    {"name": "MC + CB",           "use_ec": False, "use_cb": True,  "use_hc": False, "use_replay": False, "cb_mode": "retrained_noec"},
    {"name": "MC + EC",           "use_ec": True,  "use_cb": False, "use_hc": False, "use_replay": False, "cb_mode": "frozen_support"},
    {"name": "HC lesion",         "use_ec": True,  "use_cb": True,  "use_hc": False, "use_replay": False, "cb_mode": "frozen_with_ec"},
    {"name": "MC + EC + CB + HC", "use_ec": True,  "use_cb": True,  "use_hc": True,  "use_replay": True,  "cb_mode": "frozen_with_ec"},
    {"name": "EC lesion",         "use_ec": False, "use_cb": True,  "use_hc": True,  "use_replay": True,  "cb_mode": "retrained_noec"},
    {"name": "MTL lesion",        "use_ec": False, "use_cb": True,  "use_hc": False, "use_replay": False, "cb_mode": "frozen_zero_ec"},
]
COND_COLORS = ["dimgray", "steelblue", "seagreen", "tomato", "darkorange", "purple", "lightcoral"]
# Only run & plot: full brain, HC lesion, EC lesion, MTL lesion
KEEP_IDX = [4, 3, 5, 6]


# ── Helpers (numpy env) ──────────────────────────────────────────────────────
def phase_family(phase_idx):
    return PHASE_FAMILY[int(phase_idx)]


def env_step_np(pos, vel, action, beta, theta, gamma_curl, g_dyn, bx, by):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    curl = np.array([[0., -1.], [1., 0.]], dtype=np.float32)
    b_v  = np.array([bx, by], dtype=np.float32)
    F       = beta * (R @ action) + gamma_curl * (curl @ vel) + g_dyn * b_v
    vel_new = vel + F
    pos_new = np.clip(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new


def cosine_sim_np(a, b):
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def env_step_torch(pos, vel, action, beta, theta, gamma_curl, g_dyn, bx, by, device):
    c = torch.tensor(float(np.cos(theta)), dtype=torch.float32, device=device)
    s = torch.tensor(float(np.sin(theta)), dtype=torch.float32, device=device)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    curl = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32, device=device)
    b_v  = torch.tensor([float(bx), float(by)], dtype=torch.float32, device=device)
    F       = beta * (R @ action) + gamma_curl * (curl @ vel) + g_dyn * b_v
    vel_new = vel + F
    pos_new = torch.clamp(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new.detach()


# ── DG ────────────────────────────────────────────────────────────────────────
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


class CerebellumCB_NoEC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, 2)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

    def hidden(self, s, a):
        x = self.net[0](torch.cat([s, a], dim=-1))
        x = self.net[1](x)
        x = self.net[2](x)
        x = self.net[3](x)
        return x


class HippocampusActionRNN(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        # DG (pattern separation) — frozen random projection + top-k sparsification
        self.dg_fc = nn.Linear(LATENT_DIM, DG_DIM, bias=False)
        rng = torch.Generator()
        rng.manual_seed(seed)
        nn.init.normal_(self.dg_fc.weight, std=1.0 / LATENT_DIM ** 0.5, generator=rng)
        for p in self.dg_fc.parameters():
            p.requires_grad_(False)
        self.dg_k = max(1, int(DG_SPARSITY * DG_DIM))

        input_dim = DG_DIM + 2 + 2  # z_dg + s(pos) + g(goal)
        self.cell = nn.GRUCell(input_dim, HC_HIDDEN)
        self.readout = nn.Linear(HC_HIDDEN, 2)

    def _dg(self, z):
        """Dentate gyrus: fixed random projection + top-k sparsification."""
        h = self.dg_fc(z)
        topk_vals, _ = torch.topk(h, self.dg_k, dim=-1)
        threshold = topk_vals[..., -1:]
        return torch.relu(h - threshold)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, HC_HIDDEN, device=device)

    def step(self, z_t, s_t, g_t, h_hc):
        """z_t: EC latent (LATENT_DIM), s_t: position (2), g_t: goal (2).
        DG is applied internally to z_t."""
        z_dg_t = self._dg(z_t)
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
        if g_t.dim() == 1:
            g_t = g_t.unsqueeze(0)
        x = torch.cat([z_dg_t, s_t, g_t], dim=-1)
        h_new = self.cell(x, h_hc)
        a_hc = torch.tanh(self.readout(h_new))
        return a_hc, h_new


class MotorCortex(nn.Module):
    """
    Actor-Critic MC. Input dims vary by condition (use_ec, use_cb).
    Actor: stochastic Gaussian policy.  Critic: value head.
    """
    def __init__(self, use_ec=True, use_cb=True):
        super().__init__()
        self.use_ec = use_ec
        self.use_cb = use_cb

        input_dim = 2 + 2  # s(2) + g(2)
        if use_ec:
            input_dim += LATENT_DIM
            self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        if use_cb:
            input_dim += CB_PROJ_DIM
            self.cb_proj = nn.Linear(CB_HIDDEN, CB_PROJ_DIM, bias=False)

        self.cell         = nn.GRUCell(input_dim, MC_HIDDEN)
        self.mu_head      = nn.Linear(MC_HIDDEN, 2)
        self.log_std_head = nn.Linear(MC_HIDDEN, 2)
        self.value_head   = nn.Linear(MC_HIDDEN, 1)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def _build_input(self, s_t, h_ec, g_t, cb_hidden):
        parts = [s_t]
        if self.use_ec:
            z_ec = torch.tanh(self.ec_readout(h_ec))
            parts.append(z_ec)
        if self.use_cb and cb_hidden is not None:
            parts.append(self.cb_proj(cb_hidden))
        parts.append(g_t)
        return torch.cat(parts, dim=-1)

    def forward_step(self, s_t, h_ec, g_t, cb_hidden, h_mc):
        x     = self._build_input(s_t, h_ec, g_t, cb_hidden)
        h_new = self.cell(x, h_mc)
        mu      = self.mu_head(h_new)
        log_std = torch.clamp(self.log_std_head(h_new), ACTION_STD_MIN, ACTION_STD_MAX)
        V       = self.value_head(h_new)
        return mu, log_std, V, h_new

    def sample_action(self, s_t, h_ec, g_t, cb_hidden, h_mc):
        mu, log_std, V, h_new = self.forward_step(s_t, h_ec, g_t, cb_hidden, h_mc)
        std  = log_std.exp()
        dist = Normal(mu, std)
        u    = dist.rsample()
        a    = torch.tanh(u)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return a, log_prob, entropy, V, h_new

    def greedy_action(self, s_t, h_ec, g_t, cb_hidden, h_mc):
        mu, _, _, h_new = self.forward_step(s_t, h_ec, g_t, cb_hidden, h_mc)
        return torch.tanh(mu), h_new


# ── Replay buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=capacity)

    def push_episode(self, transitions, phase_idx, global_ep, ep_mean_zdg):
        fam = phase_family(phase_idx)
        ep_mean_zdg_np = ep_mean_zdg.detach().cpu().numpy().astype(np.float32)
        steps = []
        for tr in transitions:
            steps.append({
                "z_dg":   tr["z_dg"].detach().cpu().numpy().astype(np.float32),
                "z_cb":   tr["z_cb"].detach().cpu().numpy().astype(np.float32),
                "s":      tr["s"].copy(),
                "a":      tr["a"].detach().cpu().numpy().astype(np.float32),
                "g":      tr["g"].copy(),
            })
        self.buffer.append({
            "steps":       steps,
            "phase_idx":   int(phase_idx),
            "family":      fam,
            "global_ep":   int(global_ep),
            "ep_mean_zdg": ep_mean_zdg_np,
        })

    def __len__(self):
        return len(self.buffer)

    def sample(self, k_ep, device, current_phase_idx, current_global_ep, query_zdg_mean, phase_progress):
        """
        Gaussian-weighted sampling over the WHOLE buffer (no family split).
        Constant wide γ (REPLAY_GAMMA) → effectively uniform across the 8k buffer
        → sustained ~50/50 family balance throughout every phase.
        """
        n = len(self.buffer)
        if n == 0:
            return []

        dt = np.array(
            [max(0, current_global_ep - item["global_ep"]) for item in self.buffer],
            dtype=np.float64,
        )
        log_w = -(dt ** 2) / REPLAY_GAMMA
        log_w -= log_w.max()
        w = np.exp(log_w)
        probs = w / w.sum()

        k = min(k_ep, n)
        indices = np.random.choice(n, size=k, replace=False, p=probs).tolist()
        episodes = []
        for idx in indices:
            ep = self.buffer[idx]
            steps = []
            for s in ep["steps"]:
                steps.append({
                    "z_dg": torch.tensor(s["z_dg"], device=device).unsqueeze(0),
                    "z_cb": torch.tensor(s["z_cb"], device=device).unsqueeze(0),
                    "s":    torch.tensor(s["s"],     device=device).unsqueeze(0),
                    "a":    torch.tensor(s["a"],     device=device).unsqueeze(0),
                    "g":    torch.tensor(s["g"],     device=device).unsqueeze(0),
                })
            episodes.append(steps)
        return episodes


# ── CB-noEC retraining ────────────────────────────────────────────────────────
def sample_cb_noec_batch(batch_size, rng, device):
    s_list, a_list, sn_list = [], [], []
    ctxs = [CTX_A, CTX_B]
    for _ in range(batch_size):
        beta, theta, gamma_curl, g_dyn, bx, by, _ = ctxs[rng.integers(0, len(ctxs))]
        pos = torch.tensor(rng.uniform(0, GRID_SIZE, size=2).astype(np.float32), device=device)
        vel = torch.tensor(rng.uniform(-1.0, 1.0, size=2).astype(np.float32), device=device)
        action = torch.tensor(rng.uniform(-1.0, 1.0, size=2).astype(np.float32), device=device)
        s_next, _ = env_step_torch(pos, vel, action, beta, theta, gamma_curl, g_dyn, bx, by, device)
        s_list.append(pos); a_list.append(action); sn_list.append(s_next)
    return torch.stack(s_list), torch.stack(a_list), torch.stack(sn_list)


def get_or_train_cb_noec(device, seed=0):
    if os.path.exists(CB_NOEC_CKPT):
        print(f"Loading retrained CB_noEC from {CB_NOEC_CKPT}")
        ckpt = torch.load(CB_NOEC_CKPT, map_location=device)
        model = CerebellumCB_NoEC().to(device)
        model.load_state_dict(ckpt["cb"])
        model.eval()
        return model
    print("\nRetraining CB_noEC...")
    rng = np.random.default_rng(seed)
    model = CerebellumCB_NoEC().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_CB_NOEC)
    model.train()
    for step in range(1, CB_NOEC_STEPS + 1):
        s, a, s_next = sample_cb_noec_batch(CB_NOEC_BATCH_SIZE, rng, device)
        loss = F.mse_loss(model(s, a), s_next)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0:
            print(f"  CB_noEC step {step:4d} | loss={loss.item():.6f}")
    os.makedirs(os.path.dirname(CB_NOEC_CKPT), exist_ok=True)
    torch.save({"cb": model.state_dict()}, CB_NOEC_CKPT)
    print(f"Saved CB_noEC to {CB_NOEC_CKPT}")
    model.eval()
    return model


# ── Single episode ────────────────────────────────────────────────────────────
def run_episode(
    ec, cb_frozen, mc, device, ctx, rng,
    train=True, use_ec=True, use_cb=False, cb_noec=None,
    hc=None, use_hc=False, cb_mode="none",
    hc_pull_scale=1.0,
):
    beta, theta, gamma_curl, g_dyn, bx, by, _ = ctx

    pos_np = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal_np = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(pos_np - goal_np) < 0.5:
        goal_np = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    vel_np   = np.zeros(2, dtype=np.float32)
    g_t      = torch.tensor(goal_np, dtype=torch.float32, device=device)
    init_dist = float(np.linalg.norm(pos_np - goal_np))

    h_ec = ec.init_hidden(1, device) if ec is not None else None
    h_mc = mc.init_hidden(1, device)
    h_hc = hc.init_hidden(1, device) if (use_hc and hc is not None) else None
    z_t    = torch.zeros(1, LATENT_DIM, device=device)
    prev_a = torch.zeros(1, 2, device=device)

    rewards     = []
    log_probs   = []
    entropies   = []
    values      = []
    hc_pulls    = []
    transitions = []
    z_dg_seq    = []
    dist_seq    = []
    angle_seq   = []

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

        # CB hidden
        with torch.no_grad():
            if use_cb:
                if cb_mode == "retrained_noec":
                    cb_h = cb_noec.hidden(pos_t.unsqueeze(0), prev_a)
                elif cb_mode == "frozen_zero_ec":
                    zero_z = torch.zeros(1, LATENT_DIM, device=device)
                    cb_h = cb_frozen.hidden(pos_t.unsqueeze(0), prev_a, zero_z)
                elif cb_mode in ("frozen_with_ec", "frozen_support"):
                    cb_h = cb_frozen.hidden(pos_t.unsqueeze(0), prev_a, z_t)
                else:
                    cb_h = None
            else:
                cb_h = None

            # HC prior (DG is now inside HC)
            if use_hc and hc is not None:
                a_hc_t, h_hc = hc.step(z_t, pos_t, g_t, h_hc)
            else:
                a_hc_t = torch.zeros(1, 2, device=device)

        h_ec_mc = h_ec if (use_ec and h_ec is not None) else torch.zeros(1, EC_HIDDEN, device=device)

        if train:
            a_t, log_prob_t, entropy_t, V_t, h_mc = mc.sample_action(
                pos_t.unsqueeze(0), h_ec_mc, g_t.unsqueeze(0), cb_h, h_mc
            )
            log_probs.append(log_prob_t.squeeze(0))
            entropies.append(entropy_t.squeeze(0))
            values.append(V_t.squeeze(0).squeeze(0))

            # HC pull on MC actor (Option A: scaled by buffer composition gating)
            if use_hc and hc is not None:
                hc_pull = (hc_pull_scale * HC_ACT_WEIGHT) * torch.sum(
                    (a_t.squeeze(0) - a_hc_t.squeeze(0).detach()) ** 2
                )
                hc_pulls.append(hc_pull)
        else:
            with torch.no_grad():
                a_t, h_mc = mc.greedy_action(
                    pos_t.unsqueeze(0), h_ec_mc, g_t.unsqueeze(0), cb_h, h_mc
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

        z_dg_seq.append(z_t.squeeze(0).detach())

        transitions.append({
            "z_dg": z_t.squeeze(0),
            "z_cb":  cb_h.squeeze(0).detach() if cb_h is not None else torch.zeros(CB_HIDDEN, device=device),
            "s":    pos_np.copy(),
            "a":    a_t.squeeze(0),
            "g":    goal_np.copy(),
        })

        # Angle diagnostic
        gd    = goal_np - pos_np
        cross = gd[0] * a_np[1] - gd[1] * a_np[0]
        dot   = a_np[0] * gd[0] + a_np[1] * gd[1]
        alpha = float(np.degrees(np.arctan2(cross, dot)))
        angle_seq.append(alpha)
        if t == 0:
            first_angle = alpha

        # EC update
        if ec is not None and use_ec:
            with torch.no_grad():
                s_next_t = torch.tensor(s_next_np, dtype=torch.float32, device=device)
                s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos_t.unsqueeze(0)
                if cb_mode == "retrained_noec" and cb_noec is not None:
                    cb_pred = cb_noec(pos_t.unsqueeze(0), a_t.detach())
                else:
                    cb_pred = cb_frozen(pos_t.unsqueeze(0), a_t.detach(), z_t)
                eps = s_next_t.unsqueeze(0) - cb_pred
                z_t, h_ec = ec.step(
                    torch.cat([s_tm1, prev_a, pos_t.unsqueeze(0), eps], dim=-1), h_ec
                )

        prev_a = a_t.detach()
        pos_np = s_next_np
        vel_np = vel_new_np
        steps += 1

        if reached:
            success = True
            break

    # ── Compute A2C loss ──────────────────────────────────────────────────────
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.append(G)
    returns = list(reversed(returns))

    if train and len(returns) > 0:
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        values_t  = torch.stack(values)

        # v2 fix: per-episode advantage normalization removed — was destroying
        # the absolute scale of the success bonus and killing the gradient
        # toward better-than-current solutions.
        advantages = returns_t - values_t.detach()

        log_probs_t = torch.stack(log_probs)
        L_actor  = -(advantages * log_probs_t).mean()
        L_critic = F.mse_loss(values_t, returns_t)
        entropies_t = torch.stack(entropies)
        L_entropy = entropies_t.mean()

        mc_loss = L_actor + CRITIC_W * L_critic - ENTROPY_W * L_entropy

        # Add HC pull if present
        if len(hc_pulls) > 0:
            mc_loss = mc_loss + torch.stack(hc_pulls).mean()

        adv_std = float(advantages.std().item()) if len(advantages) > 1 else 0.0
    else:
        mc_loss   = torch.tensor(0.0, device=device, requires_grad=True)
        L_actor   = mc_loss
        L_critic  = mc_loss
        adv_std   = 0.0

    ep_mean_zdg = (
        torch.stack(z_dg_seq).mean(dim=0)
        if len(z_dg_seq) > 0 else torch.zeros(DG_DIM, device=device)
    )

    ep_return   = float(sum(rewards))
    dist_norm   = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
    mean_angle  = float(np.mean(angle_seq)) if angle_seq else 0.0
    first_angle = first_angle if first_angle is not None else 0.0

    return {
        "mc_loss":      mc_loss,
        "L_actor":      float(L_actor.detach().item()) if train else 0.0,
        "L_critic":     float(L_critic.detach().item()) if train else 0.0,
        "adv_std":      adv_std,
        "success":      float(success),
        "steps":        steps,
        "dist_norm":    dist_norm,
        "angle":        mean_angle,
        "first_angle":  first_angle,
        "ep_return":    ep_return,
        "transitions":  transitions,
        "ep_mean_zdg":  ep_mean_zdg.detach(),
    }


# ── HC replay train ───────────────────────────────────────────────────────────
def hc_replay_train(hc, replay_buffer, hc_optimizer, device,
                    current_phase_idx, current_global_ep,
                    query_zdg_mean, phase_progress):
    if hc is None or replay_buffer is None or hc_optimizer is None:
        return 0.0
    if len(replay_buffer) < REPLAY_MIN_EP:
        return 0.0

    episodes = replay_buffer.sample(
        REPLAY_K_EP, device,
        current_phase_idx=current_phase_idx,
        current_global_ep=current_global_ep,
        query_zdg_mean=query_zdg_mean,
        phase_progress=phase_progress,
    )
    if len(episodes) == 0:
        return 0.0

    hc.train()
    hc_optimizer.zero_grad()

    total_loss = torch.tensor(0.0, device=device)
    n_steps = 0
    for ep_steps in episodes:
        h_hc = hc.init_hidden(1, device)
        for step in ep_steps:
            a_pred, h_hc = hc.step(step["z_dg"], step["s"].squeeze(0), step["g"].squeeze(0), h_hc)
            total_loss = total_loss + F.mse_loss(a_pred, step["a"])
            n_steps += 1

    loss = total_loss / max(n_steps, 1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(hc.parameters(), 1.0)
    hc_optimizer.step()
    hc.eval()
    return float(loss.item())


# ── Training loop ─────────────────────────────────────────────────────────────
def run_babab(ec, cb_frozen, mc, device, seed=0,
              use_ec=True, use_cb=False, cb_noec=None,
              use_hc=False, use_replay=False, cb_mode="none"):
    ec.eval(); cb_frozen.eval()
    for p in ec.parameters():       p.requires_grad_(False)
    for p in cb_frozen.parameters(): p.requires_grad_(False)

    hc = HippocampusActionRNN(seed=0).to(device) if use_hc else None
    hc_optimizer = torch.optim.Adam(hc.parameters(), lr=LR_HC) if hc is not None else None
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE) if use_replay else None

    mc_optimizer = torch.optim.Adam(mc.parameters(), lr=LR_MC)
    rng = np.random.default_rng(seed)

    curve = {
        "ep": [], "success": [], "steps": [], "dist_norm": [],
        "angle": [], "first_angle": [], "phase": [], "return": [],
        "L_actor": [], "L_critic": [], "L_total": [], "adv_std": [],
        "hc_loss": [],
    }

    global_ep = 0

    for phase_idx, ctx in enumerate(PHASES):
        print(f"\n  Phase {PHASE_LABELS[phase_idx]} ({N_PHASE_EP} episodes)...")
        phase_start_ep = global_ep  # snapshot before this phase's episodes

        mc_optimizer.zero_grad()
        for ep in range(1, N_PHASE_EP + 1):
            global_ep += 1

            # HC pull ramp: scales from 0 → 1 as same-family episodes
            # accumulate in the buffer (0 → HC_PULL_GATE_N). Acts in ALL
            # phases. With 50/50 buffer balance, HC trains on same-family
            # episodes from ep 1 → its suggestions are relevant, not just
            # from the previous (wrong) family.
            if use_hc and use_replay and replay_buffer is not None:
                cur_fam = phase_family(phase_idx)
                n_same = sum(1 for it in replay_buffer.buffer if it["family"] == cur_fam)
                hc_pull_scale = min(1.0, n_same / float(HC_PULL_GATE_N))
            else:
                hc_pull_scale = 1.0

            mc.train()
            out = run_episode(
                ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
                ctx=ctx, rng=rng, train=True,
                use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec,
                hc=hc, use_hc=use_hc, cb_mode=cb_mode,
                hc_pull_scale=hc_pull_scale,
            )

            # Accumulate gradients, update every BATCH_EPS
            (out["mc_loss"] / BATCH_EPS).backward()
            if ep % BATCH_EPS == 0:
                torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
                mc_optimizer.step()
                mc_optimizer.zero_grad()

            # Push to replay buffer
            if use_replay and replay_buffer is not None:
                replay_buffer.push_episode(
                    transitions=out["transitions"],
                    phase_idx=phase_idx,
                    global_ep=global_ep,
                    ep_mean_zdg=out["ep_mean_zdg"],
                )

            # HC replay training
            # phase_progress = local episode count within current phase (≥ 0).
            # Used by ReplayBuffer.sample() to schedule γ (wide at boundary,
            # narrow as the phase matures).
            hc_loss = hc_replay_train(
                hc=hc, replay_buffer=replay_buffer,
                hc_optimizer=hc_optimizer, device=device,
                current_phase_idx=phase_idx,
                current_global_ep=global_ep,
                query_zdg_mean=out["ep_mean_zdg"],
                phase_progress=float(ep),
            )

            # Eval every 10 episodes (greedy, no noise)
            if ep % 10 == 0:
                mc.eval()
                with torch.no_grad():
                    eval_out = run_episode(
                        ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
                        ctx=ctx, rng=rng, train=False,
                        use_ec=use_ec, use_cb=use_cb, cb_noec=cb_noec,
                        hc=hc, use_hc=use_hc, cb_mode=cb_mode
                    )
                curve["ep"].append(global_ep)
                curve["success"].append(float(eval_out["success"]))
                curve["steps"].append(float(eval_out["steps"]))
                curve["dist_norm"].append(float(eval_out["dist_norm"]))
                curve["angle"].append(float(eval_out["angle"]))
                curve["first_angle"].append(float(eval_out["first_angle"]))
                curve["phase"].append(phase_idx)
                curve["return"].append(float(eval_out["ep_return"]))
                curve["L_actor"].append(out["L_actor"])
                curve["L_critic"].append(out["L_critic"])
                curve["L_total"].append(float(out["mc_loss"].detach().item()))
                curve["adv_std"].append(out["adv_std"])
                curve["hc_loss"].append(float(hc_loss))

            if ep % 200 == 0:
                w = min(20, len(curve['success']))
                print(
                    f"    ep {ep:4d} | "
                    f"succ={np.mean(curve['success'][-w:]):.2f} | "
                    f"ret={np.mean(curve['return'][-w:]):.1f} | "
                    f"L_tot={np.mean(curve['L_total'][-w:]):.1f} | "
                    f"hc={np.mean(curve['hc_loss'][-w:]):.3f}",
                    flush=True
                )

    return curve


# ── Plot helpers ──────────────────────────────────────────────────────────────
EVAL_EVERY = 10
N_EVAL_PER_PHASE = N_PHASE_EP // EVAL_EVERY
SW = 20
ep_local = np.arange(1, N_EVAL_PER_PHASE + 1) * EVAL_EVERY


def smooth(x, w=SW):
    x = np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x


def exp_model(xv, L, y0, k):
    return L + (y0 - L) * np.exp(-k * xv)


def fit_asymptote(x, y):
    try:
        x, y = np.asarray(x, float), np.asarray(y, float)
        popt, _ = curve_fit(exp_model, x, y, p0=[float(y[-1]), float(y[0]), 1e-3], maxfev=5000)
        L, y0, k = popt
        x95 = -np.log(0.05) / k if (k > 0 and abs(L - y0) > 1e-6) else None
        return L, y0, k, x95
    except Exception:
        return None, None, None, None


def split_phases(arr):
    arr = np.array(arr)
    return [arr[i * N_EVAL_PER_PHASE:(i + 1) * N_EVAL_PER_PHASE] for i in range(len(PHASES))]


def plot_seed(seed, curves_all, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    curves      = [curves_all[i] for i in KEEP_IDX]
    cond_names  = [CONDITIONS[i]["name"] for i in KEEP_IDX]
    cond_colors = [COND_COLORS[i] for i in KEEP_IDX]
    n_conds     = len(curves)

    intact_name   = "MC + EC + CB + HC"
    lesion_name   = "MTL lesion"
    name_to_curve = {CONDITIONS[i]["name"]: curves_all[i] for i in range(len(CONDITIONS))}
    name_to_color = {CONDITIONS[i]["name"]: COND_COLORS[i] for i in range(len(CONDITIONS))}
    EP_SHOW = 800

    phase_bounds = [i * N_EVAL_PER_PHASE for i in range(1, len(PHASES))]

    # Fig 1: Learning curves (success)
    fig, axes = plt.subplots(n_conds, 1, figsize=(14, 3 * n_conds), squeeze=False)
    for row, (name, color, curve) in enumerate(zip(cond_names, cond_colors, curves)):
        ax = axes[row, 0]
        succ = np.array(curve["success"])
        y_sm = smooth(succ)
        ep_g = np.arange(1, len(succ) + 1)
        ax.plot(ep_g[SW - 1:], y_sm, color=color, lw=1.5)
        for xb in phase_bounds:
            ax.axvline(xb, color="gray", ls="--", lw=0.8)
        for i, lbl in enumerate(PHASE_LABELS):
            ax.text((i + 0.5) * N_EVAL_PER_PHASE, 1.02, lbl, ha="center", fontsize=7,
                    transform=ax.get_xaxis_transform())
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(name, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if row == n_conds - 1:
            ax.set_xlabel("Episode (global)")
    fig.suptitle(f"A2C Learning Curves — all conditions (seed {seed})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2: A vs A2 success overlay
    fig, axes = plt.subplots(n_conds, 1, figsize=(8, 3 * n_conds), squeeze=False)
    for row, (name, color, curve) in enumerate(zip(cond_names, cond_colors, curves)):
        ax = axes[row, 0]
        phases = split_phases(curve["success"])
        for ph_idx, col, lbl in [
            (A_IDX,  "steelblue",  "A1"), (A2_IDX, "darkorange", "A2"),
        ]:
            y_sm = smooth(np.array(phases[ph_idx]))
            ep_sm = ep_local[SW - 1:]
            mask = ep_sm <= EP_SHOW
            ax.plot(ep_sm[mask], y_sm[mask], color=col, lw=2, label=lbl)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylabel(f"{name}\nSuccess", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if row == n_conds - 1:
            ax.set_xlabel("Episode within phase")
    fig.suptitle(f"A1 vs A2 success — savings (seed {seed})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/a1_vs_a2_success.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3: A vs A2 angle overlay
    fig, axes = plt.subplots(n_conds, 1, figsize=(8, 3 * n_conds), squeeze=False)
    for row, (name, color, curve) in enumerate(zip(cond_names, cond_colors, curves)):
        ax = axes[row, 0]
        phases = split_phases(curve["angle"])
        for ph_idx, col, lbl in [
            (A_IDX,  "steelblue",  "A1"), (A2_IDX, "darkorange", "A2"),
        ]:
            y_sm = smooth(np.array(phases[ph_idx]))
            ep_sm = ep_local[SW - 1:]
            mask = ep_sm <= EP_SHOW
            ax.plot(ep_sm[mask], y_sm[mask], color=col, lw=2, label=lbl)
        ax.axhline(-60, color="gray", ls="--", lw=1.2, label="-60° target")
        ax.set_xlim(1, EP_SHOW); ax.set_ylim(-110, 60)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylabel(f"{name}\nAngle (deg)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if row == n_conds - 1:
            ax.set_xlabel("Episode within phase")
    fig.suptitle(f"A1 vs A2 angle (seed {seed})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/angle_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 4: Savings bar
    fig, ax = plt.subplots(figsize=(10, 4))
    x95_A, x95_A2 = [], []
    for curve in curves:
        phases = split_phases(curve["success"])
        for ph_idx, store in [(A_IDX, x95_A), (A2_IDX, x95_A2)]:
            y_sm = smooth(np.array(phases[ph_idx]))
            ep_sm = ep_local[SW - 1:]
            _, _, _, x95 = fit_asymptote(ep_sm, y_sm)
            store.append(x95 if (x95 is not None and x95 < N_EVAL_PER_PHASE) else N_EVAL_PER_PHASE)
    width = 0.35
    xs = np.arange(n_conds)
    for i, (name, color, a, a2) in enumerate(zip(cond_names, cond_colors, x95_A, x95_A2)):
        norm_a2 = a2 / a if a > 0 else 1.0
        ax.bar(i - width / 2, 1.0, width, color=color, alpha=0.4)
        ax.bar(i + width / 2, norm_a2, width, color=color, alpha=1.0)
        ax.annotate("", xy=(i + width / 2, norm_a2), xytext=(i + width / 2, 1.0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.text(i + width / 2 + 0.08, (1.0 + norm_a2) / 2, f"{norm_a2:.2f}", fontsize=8, va="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(cond_names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Episodes to 95% asymptote\n(normalised to A = 1.0)")
    ax.set_title(f"Savings: A vs A2 (seed {seed})")
    ax.set_ylim(0, 1.5)
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="A1"),
                       Patch(facecolor="gray", alpha=1.0, label="A2")], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/savings_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 5: Diagnostic losses (total MC loss per condition)
    fig, axes = plt.subplots(n_conds, 1, figsize=(14, 3 * n_conds), squeeze=False)
    for row, (name, color, curve) in enumerate(zip(cond_names, cond_colors, curves)):
        ax = axes[row, 0]
        y = smooth(np.array(curve["L_total"]))
        ep_g = np.arange(1, len(curve["L_total"]) + 1)
        ax.plot(ep_g[SW - 1:], y, color=color, lw=1.5)
        ax.axhline(0, color="black", ls=":", lw=0.7)
        for xb in phase_bounds:
            ax.axvline(xb, color="gray", ls="--", lw=0.8)
        for i, lbl in enumerate(PHASE_LABELS):
            ax.text((i + 0.5) * N_EVAL_PER_PHASE, 1.02, lbl, ha="center", fontsize=7,
                    transform=ax.get_xaxis_transform())
        ax.set_ylabel(f"{name}\nTotal MC loss", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if row == n_conds - 1:
            ax.set_xlabel("Episode (global)")
    fig.suptitle(f"Total MC loss — all conditions (seed {seed})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/mc_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 6: Intact vs MTL lesion
    if intact_name in name_to_curve and lesion_name in name_to_curve:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, ph_idx, ph_lbl in [(axes[0], A_IDX, PHASE_LABELS[A_IDX]),
                                   (axes[1], A2_IDX, PHASE_LABELS[A2_IDX])]:
            for name, ls, lw, alpha in [(intact_name, "-", 2.0, 1.0), (lesion_name, "--", 2.0, 0.8)]:
                phases = split_phases(name_to_curve[name]["success"])
                y_sm = smooth(np.array(phases[ph_idx]))
                ep_sm = ep_local[SW - 1:]
                mask = ep_sm <= EP_SHOW
                ax.plot(ep_sm[mask], y_sm[mask], color=name_to_color[name],
                        lw=lw, alpha=alpha, ls=ls, label=name)
            ax.set_ylim(0, 1.1)
            ax.set_title(ph_lbl)
            ax.set_xlabel("Episode within phase")
            ax.set_ylabel("Success")
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Intact vs MTL lesion (seed {seed})")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/mtl_lesion.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Fig 7: ABA grid
    ABA_PHASE_INDICES = [A_IDX, B_IDX, A2_IDX]
    ABA_PHASE_LABELS  = [PHASE_LABELS[i] for i in ABA_PHASE_INDICES]
    fig, axes = plt.subplots(n_conds, 3, figsize=(14, 3 * n_conds), squeeze=False)
    for row, (name, color, curve) in enumerate(zip(cond_names, cond_colors, curves)):
        succ_phases = split_phases(curve["success"])
        for col, ph_idx in enumerate(ABA_PHASE_INDICES):
            ax = axes[row, col]
            y_raw = np.array(succ_phases[ph_idx])
            y_sm = smooth(y_raw)
            ep_sm = ep_local[SW - 1:]
            ax.plot(ep_sm, y_sm, color=color, lw=1.8)
            L, y0, k, x95 = fit_asymptote(ep_sm, y_sm)
            if L is not None:
                ax.plot(ep_sm, exp_model(ep_sm, L, y0, k),
                        color="black", lw=1.2, ls="--", alpha=0.7)
                if x95 is not None and x95 < N_EVAL_PER_PHASE:
                    ax.axvline(x95, color="black", lw=1, ls=":", alpha=0.6)
                    ax.text(x95 + 20, 0.1, f"x95={x95:.0f}", fontsize=6)
            ax.set_ylim(0, 1.1)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(ABA_PHASE_LABELS[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{name}\nSuccess", fontsize=7)
            if row == n_conds - 1:
                ax.set_xlabel("Episode within phase")
    fig.suptitle(f"ABA accuracy + asymptote fit (seed {seed})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/aba_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 8: Per-seed slope comparison
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(n_conds)
    width = 0.35
    for i, (name, color, curve) in enumerate(zip(cond_names, cond_colors, curves)):
        phases = split_phases(curve["success"])
        for ph_idx, tag, offset, alpha in [(A_IDX, "A1", -width/2, 0.4), (A2_IDX, "A2", width/2, 1.0)]:
            y = np.array(phases[ph_idx][:SLOPE_WINDOW], dtype=float)
            if len(y) == SLOPE_WINDOW:
                slope = np.polyfit(np.arange(SLOPE_WINDOW, dtype=float), y, 1)[0] * 1000
            else:
                slope = 0.0
            ax.bar(i + offset, slope, width, color=color, alpha=alpha)
    ax.set_xticks(xs)
    ax.set_xticklabels(cond_names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Learning slope (Δsuccess / 1000 eps)")
    ax.set_title(f"Initial learning rate: A1 (light) vs A2 (dark) — first {SLOPE_WINDOW} eps (seed {seed})")
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="A1"),
                       Patch(facecolor="gray", alpha=1.0, label="A2")], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/savings_slope.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 9: Per-seed early performance
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (name, color, curve) in enumerate(zip(cond_names, cond_colors, curves)):
        phases = split_phases(curve["success"])
        for ph_idx, tag, offset, alpha in [(A_IDX, "A1", -width/2, 0.4), (A2_IDX, "A2", width/2, 1.0)]:
            early = float(np.mean(np.array(phases[ph_idx][:SLOPE_WINDOW])))
            ax.bar(i + offset, early, width, color=color, alpha=alpha)
    ax.set_xticks(xs)
    ax.set_xticklabels(cond_names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(f"Mean success (first {SLOPE_WINDOW} eps)")
    ax.set_title(f"Early A-phase performance: A1 vs A2 (seed {seed})")
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="A1"),
                       Patch(facecolor="gray", alpha=1.0, label="A2")], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/savings_early.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Seed {seed}: figures saved to {fig_dir}")


# ── Cross-seed savings analysis ──────────────────────────────────────────────
SLOPE_WINDOW = 50    # first N eval points (= 500 train episodes) to fit slope


def compute_savings_metrics(curves_all_seeds):
    """
    For each condition × seed, compute:
      - slope_A1, slope_A2: linear slope over first SLOPE_WINDOW episodes
      - auc_A1, auc_A2: area under curve (mean success) over full phase
      - early_A1, early_A2: mean success over first SLOPE_WINDOW episodes
    Returns dict[cond_idx] -> dict of lists across seeds.
    """
    n_conds = len(CONDITIONS)
    metrics = {i: {"slope_A1": [], "slope_A2": [], "auc_A1": [], "auc_A2": [],
                    "early_A1": [], "early_A2": []} for i in range(n_conds)}

    for seed_curves in curves_all_seeds:
        for ci, curve in enumerate(seed_curves):
            phases = split_phases(curve["success"])
            for ph_idx, tag in [(A_IDX, "A1"), (A2_IDX, "A2")]:
                y = np.array(phases[ph_idx], dtype=float)
                # Early mean
                early = float(np.mean(y[:SLOPE_WINDOW]))
                metrics[ci][f"early_{tag}"].append(early)
                # AUC (mean success over full phase)
                metrics[ci][f"auc_{tag}"].append(float(np.mean(y)))
                # Slope: linear fit over first SLOPE_WINDOW episodes
                x_fit = np.arange(SLOPE_WINDOW, dtype=float)
                y_fit = y[:SLOPE_WINDOW]
                if len(y_fit) == SLOPE_WINDOW:
                    coeffs = np.polyfit(x_fit, y_fit, 1)
                    metrics[ci][f"slope_{tag}"].append(float(coeffs[0]))
                else:
                    metrics[ci][f"slope_{tag}"].append(0.0)
    return metrics


def plot_cross_seed_savings(curves_all_seeds, fig_dir):
    """Plot savings metrics across available seeds."""
    os.makedirs(fig_dir, exist_ok=True)
    n_seeds = len(curves_all_seeds)
    if n_seeds == 0:
        return

    metrics = compute_savings_metrics(curves_all_seeds)
    cond_names  = [CONDITIONS[i]["name"] for i in KEEP_IDX]
    cond_colors = [COND_COLORS[i] for i in KEEP_IDX]
    n_conds = len(KEEP_IDX)

    # ── Fig 1: A1 vs A2 learning curves (mean ± SEM across seeds) ────────────
    fig, axes = plt.subplots(n_conds, 1, figsize=(8, 3 * n_conds), squeeze=False)
    for row, ci in enumerate(KEEP_IDX):
        ax = axes[row, 0]
        for ph_idx, col, lbl in [(A_IDX, "steelblue", "A1"), (A2_IDX, "darkorange", "A2")]:
            all_curves = []
            for seed_curves in curves_all_seeds:
                phases = split_phases(seed_curves[ci]["success"])
                all_curves.append(np.array(phases[ph_idx], dtype=float))
            stacked = np.stack(all_curves)  # (n_seeds, N_PHASE_EP)
            # Smooth each seed then average
            sw = 40
            smoothed = np.array([np.convolve(s, np.ones(sw)/sw, mode="valid") for s in stacked])
            mean_y = smoothed.mean(axis=0)
            sem_y  = smoothed.std(axis=0) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(mean_y)
            ep = np.arange(sw, sw + len(mean_y))
            ax.plot(ep, mean_y, color=col, lw=2, label=lbl)
            ax.fill_between(ep, mean_y - sem_y, mean_y + sem_y, color=col, alpha=0.2)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(f"{cond_names[row]}\nSuccess", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        if row == n_conds - 1:
            ax.set_xlabel("Episode within phase")
    fig.suptitle(f"A1 vs A2 mean ± SEM ({n_seeds} seeds)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/savings_a1_vs_a2_mean.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 2: Slope comparison bar chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(n_conds)
    width = 0.35
    for i, ci in enumerate(KEEP_IDX):
        s_a1 = np.array(metrics[ci]["slope_A1"]) * 1000  # slope per 1000 eps
        s_a2 = np.array(metrics[ci]["slope_A2"]) * 1000
        ax.bar(i - width/2, s_a1.mean(), width, color=cond_colors[i], alpha=0.4,
               yerr=s_a1.std()/np.sqrt(n_seeds) if n_seeds > 1 else 0, capsize=3)
        ax.bar(i + width/2, s_a2.mean(), width, color=cond_colors[i], alpha=1.0,
               yerr=s_a2.std()/np.sqrt(n_seeds) if n_seeds > 1 else 0, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(cond_names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Learning slope (Δsuccess / 1000 eps)")
    ax.set_title(f"Initial learning rate: A1 (light) vs A2 (dark) — first {SLOPE_WINDOW} eps ({n_seeds} seeds)")
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="A1"),
                       Patch(facecolor="gray", alpha=1.0, label="A2")], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/savings_slope.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Fig 3: Early performance bar chart ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, ci in enumerate(KEEP_IDX):
        e_a1 = np.array(metrics[ci]["early_A1"])
        e_a2 = np.array(metrics[ci]["early_A2"])
        ax.bar(i - width/2, e_a1.mean(), width, color=cond_colors[i], alpha=0.4,
               yerr=e_a1.std()/np.sqrt(n_seeds) if n_seeds > 1 else 0, capsize=3)
        ax.bar(i + width/2, e_a2.mean(), width, color=cond_colors[i], alpha=1.0,
               yerr=e_a2.std()/np.sqrt(n_seeds) if n_seeds > 1 else 0, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(cond_names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(f"Mean success (first {SLOPE_WINDOW} eps)")
    ax.set_title(f"Early A-phase performance: A1 vs A2 ({n_seeds} seeds)")
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.4, label="A1"),
                       Patch(facecolor="gray", alpha=1.0, label="A2")], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/savings_early.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Cross-seed savings plots saved to {fig_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = DEVICE
    print(f"Device: {device}")

    ckpt = torch.load(CKPT, map_location=device)
    ec = EntorhinalCortex().to(device)
    cb_frozen = CerebellumCB().to(device)
    ec.load_state_dict(ckpt["ec"])
    cb_frozen.load_state_dict(ckpt["cb"])
    ec.eval(); cb_frozen.eval()
    print(f"Loaded EC + CB from {CKPT}")

    cb_noec_master = get_or_train_cb_noec(device=device, seed=0)

    for seed in [0]:   # v2 first pass: seed 0 only
        seed_dir = os.path.join(FIG_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        curves_all = [None] * len(CONDITIONS)
        for cond_idx in KEEP_IDX:
            cond = CONDITIONS[cond_idx]
            cond_path = os.path.join(seed_dir, f"cond_{cond_idx}.npz")
            if os.path.exists(cond_path):
                print(f"Condition {cond['name']}: cached, loading.")
                d = np.load(cond_path, allow_pickle=True)
                curves_all[cond_idx] = d["curve"].item()
                continue

            print(f"\n{'=' * 60}")
            print(f"Seed {seed} | Condition: {cond['name']}")
            print(f"{'=' * 60}")

            mc = MotorCortex(use_ec=cond["use_ec"], use_cb=cond["use_cb"]).to(device)

            cb_noec = None
            if cond["cb_mode"] == "retrained_noec":
                cb_noec = CerebellumCB_NoEC().to(device)
                cb_noec.load_state_dict(cb_noec_master.state_dict())
                cb_noec.eval()
                for p in cb_noec.parameters():
                    p.requires_grad_(False)

            curve = run_babab(
                ec=ec, cb_frozen=cb_frozen, mc=mc, device=device,
                seed=seed,
                use_ec=cond["use_ec"], use_cb=cond["use_cb"],
                cb_noec=cb_noec,
                use_hc=cond["use_hc"], use_replay=cond["use_replay"],
                cb_mode=cond["cb_mode"],
            )
            curves_all[cond_idx] = curve
            np.savez(cond_path, curve=curve)
            print(f"  Saved condition {cond_idx} to {cond_path}")

        np.savez(os.path.join(seed_dir, "all_conditions_data.npz"), curves=curves_all)
        plot_seed(seed, curves_all, seed_dir)
        print(f"Seed {seed} done.")

    os.makedirs(FIG_DIR, exist_ok=True)
    summary = {str(i): cond for i, cond in enumerate(CONDITIONS)}
    with open(os.path.join(FIG_DIR, "conditions.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

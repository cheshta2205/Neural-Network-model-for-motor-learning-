"""
structure_learning_drift.py
───────────────────────────
Section 3 (structure learning), DRIFT family — parallel to structure_learning_rot.py.

Question: if EC→CB is trained on only a few drift magnitudes, does it learn the
drift *axis* (drift strength as a continuous variable) and generalize to UNSEEN
magnitudes?

Setup
-----
  • Train EC→CB (MSE next-state prediction) on drift magnitudes d ∈ {0.4, 0.8, 1.2}.
  • Drift is a constant force along +x:  F = action + d·[1, 0]   (no rotation/curl).
  • Test on a dense sweep of held-out magnitudes (interpolation inside 0.4–1.2,
    extrapolation below/above, incl. d=0 = no drift).
  • Plot CB prediction MSE vs drift magnitude. Flat-low across the trained range
    = learned the drift axis; spikes only at trained values = memorized 3 contexts.
  • Bonus: fit a linear d-decoder on the 3 TRAINED magnitudes' z_ec and show it
    predicts the correct magnitude for held-out contexts (metric axis).

Drift magnitude is a non-periodic axis, so extrapolation is cleaner than rotation.
No MC / SAC / replay buffer — pure forward model (supervised MSE).
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

try:
    from src.PlottingTools.plotting_functions import set_plt_params, make_axis_nice
    _HAS_PLOT_HELPERS = True
except Exception:                                    # pragma: no cover
    _HAS_PLOT_HELPERS = False
    def set_plt_params(*a, **k): pass
    def make_axis_nice(*a, **k): pass


# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_DRIFTS = [0.4, 0.8, 1.2]                        # 3 anchors (magnitude along +x)
TEST_DRIFTS  = np.round(np.arange(0.0, 1.61, 0.05), 3)  # dense sweep (incl. anchors)

SEQ_LEN    = 80
N_TRAIN    = 12000
N_TEST_PER_DRIFT = 200
BATCH_SIZE = 256
EPOCHS     = 50
LR         = 1e-3
WARMUP     = 10        # skip first WARMUP steps in MSE (let EC settle)

LATENT_DIM = 64
EC_HIDDEN  = 256
CB_HIDDEN  = 256
EC_INPUT   = 8
GRID_SIZE  = 4
DT         = 1.0

FIG_DIR   = "paper/structure_learning"


def _tag():
    """Filename tag from the current training-drift set, e.g. '0p4_0p8_1p2'."""
    return "_".join(f"{d:g}".replace(".", "p") for d in TRAIN_DRIFTS)


def ckpt_path():
    return f"paper/checkpoints/structure_learning_drift_{_tag()}.pt"


# ── Drift-only environment step ─────────────────────────────────────────────
def env_step(pos, vel, action, drift):
    """One velocity-based step under a constant +x drift:  F = action + d·[1,0]."""
    f = action + np.array([drift, 0.0], dtype=np.float32)
    vel_new = vel + f
    pos_new = np.clip(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new.astype(np.float32), vel_new.astype(np.float32)


# ── Data: constant-context sequences at given drift magnitudes ──────────────
def generate_sequences(drifts, n_seqs, seq_len=SEQ_LEN, seed=0):
    """Each sequence runs a single (constant) drift magnitude drawn from `drifts`.
    Returns transitions (N,T,6) and the drift magnitude per sequence (N,)."""
    rng = np.random.default_rng(seed)
    drifts = np.asarray(drifts, dtype=np.float64)

    all_trans, all_d = [], []
    for _ in range(n_seqs):
        d = float(rng.choice(drifts))
        pos = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
        vel = np.zeros(2, dtype=np.float32)

        seq = []
        for _t in range(seq_len):
            action = rng.uniform(-1, 1, size=2).astype(np.float32)
            pos_next, vel = env_step(pos, vel, action, d)
            seq.append([pos[0], pos[1], action[0], action[1],
                        pos_next[0], pos_next[1]])
            pos = pos_next
        all_trans.append(seq)
        all_d.append(d)

    return (torch.tensor(np.array(all_trans), dtype=torch.float32),
            np.array(all_d, dtype=np.float32))


class TransitionDataset(Dataset):
    def __init__(self, transitions): self.t = transitions
    def __len__(self): return len(self.t)
    def __getitem__(self, i): return self.t[i]


# ── Models (same architecture as the EC→CB forward model) ───────────────────
class EntorhinalCortex(nn.Module):
    def __init__(self, input_dim=EC_INPUT, hidden_dim=EC_HIDDEN, latent_dim=LATENT_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell    = nn.GRUCell(input_dim, hidden_dim)
        self.readout = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.Tanh())

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(self, x, h):
        h_new = self.cell(x, h)
        return self.readout(h_new), h_new


class CerebellumCB(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden=CB_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),         nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, s, a, z):
        return self.net(torch.cat([s, a, z], dim=-1))


# ── Closed-loop rollout (EC infers context from CB prediction error) ────────
def run_sequence(ec, cb, trans, device):
    B, T, _ = trans.shape
    zeros2 = torch.zeros(B, 2, device=device)
    h      = ec.init_hidden(B, device)
    s_pred = zeros2.clone()

    z_seq, s_pred_seq = [], []
    for t in range(T):
        s_t = trans[:, t, :2]
        a_t = trans[:, t, 2:4]
        if t == 0:
            s_tm1 = a_tm1 = e_t = zeros2
        else:
            s_tm1 = trans[:, t-1, :2]
            a_tm1 = trans[:, t-1, 2:4]
            e_t   = s_t - s_pred

        z_t, h = ec.step(torch.cat([s_tm1, a_tm1, s_t, e_t], dim=-1), h)
        s_pred = cb(s_t, a_t, z_t)
        z_seq.append(z_t); s_pred_seq.append(s_pred)

    return torch.stack(z_seq, 1), torch.stack(s_pred_seq, 1)


# ── Train ───────────────────────────────────────────────────────────────────
def train(seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | train drifts: {TRAIN_DRIFTS}", flush=True)

    tr_trans, _ = generate_sequences(TRAIN_DRIFTS, N_TRAIN, seed=seed)
    loader = DataLoader(TransitionDataset(tr_trans), batch_size=BATCH_SIZE, shuffle=True)

    ec = EntorhinalCortex().to(device)
    cb = CerebellumCB().to(device)
    opt = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        ec.train(); cb.train(); tot = 0.0
        for trans in loader:
            trans = trans.to(device)
            _, s_pred = run_sequence(ec, cb, trans, device)
            loss = F.mse_loss(s_pred, trans[:, :, 4:6])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(trans)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:03d} | train MSE {tot / N_TRAIN:.4f}", flush=True)

    os.makedirs(os.path.dirname(ckpt_path()), exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "seed": seed}, ckpt_path())
    print(f"Saved {ckpt_path()}", flush=True)
    return ec, cb, device


def load(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path(), map_location=device)
    ec = EntorhinalCortex().to(device); ec.load_state_dict(ck["ec"]); ec.eval()
    cb = CerebellumCB().to(device);     cb.load_state_dict(ck["cb"]); cb.eval()
    return ec, cb, device


# ── Generalization sweep over held-out drift magnitudes ─────────────────────
def sweep(ec, cb, device, seed=999):
    ec.eval(); cb.eval()
    mse_mean, mse_sem, z_by_drift = [], [], []
    with torch.no_grad():
        for d in TEST_DRIFTS:
            trans, _ = generate_sequences([d], N_TEST_PER_DRIFT, seed=seed + int(d * 100))
            trans = trans.to(device)
            z, s_pred = run_sequence(ec, cb, trans, device)
            target = trans[:, :, 4:6]
            se = (s_pred[:, WARMUP:] - target[:, WARMUP:]) ** 2
            per_seq = se.mean(dim=(1, 2)).cpu().numpy()
            mse_mean.append(per_seq.mean())
            mse_sem.append(per_seq.std() / np.sqrt(len(per_seq)))
            z_by_drift.append(z[:, WARMUP:].mean(dim=1).cpu().numpy())
    return np.array(mse_mean), np.array(mse_sem), z_by_drift


def drift_decode(z_by_drift):
    """Fit linear z_ec→d on the TRAINED magnitudes only, predict for all test drifts."""
    train_set = set(np.round(TRAIN_DRIFTS, 3))
    Xtr, ytr = [], []
    for d, Z in zip(TEST_DRIFTS, z_by_drift):
        if round(float(d), 3) in train_set:
            Xtr.append(Z); ytr.append(np.full(len(Z), d))
    if not Xtr:
        return None
    reg = LinearRegression().fit(np.vstack(Xtr), np.concatenate(ytr))
    return np.array([reg.predict(Z).mean() for Z in z_by_drift])


# ── Baselines + matched oracle (parallel to structure_learning_rot.py) ──────
def _dnorm():
    return float(max(TRAIN_DRIFTS))


def _drift_context(drifts, condition):
    drifts = np.asarray(drifts, dtype=np.float64)
    if condition == "scalar":
        return (drifts / _dnorm()).reshape(-1, 1).astype(np.float32)
    if condition == "onehot":
        train = np.asarray(TRAIN_DRIFTS, dtype=np.float64)
        idx = np.argmin(np.abs(drifts[:, None] - train[None, :]), axis=1)   # nearest trained
        return np.eye(len(train), dtype=np.float32)[idx]
    raise ValueError(condition)


def _with_prev(trans):
    s_t    = trans[:, :, :2]
    s_prev = torch.cat([s_t[:, :1], s_t[:, :-1]], dim=1)
    return s_prev, s_t, trans[:, :, 2:4], trans[:, :, 4:6]


class ContextCB(nn.Module):
    """CB given an EXPLICIT context vector (no EC). Input: [s_{t-1}, s_t, a_t, ctx]."""
    def __init__(self, ctx_dim, hidden=CB_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6 + ctx_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),      nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, s_prev, s_t, a_t, c):
        return self.net(torch.cat([s_prev, s_t, a_t, c], dim=-1))


def train_context_cb(condition, device, epochs=20, seed=0):
    torch.manual_seed(seed)
    trans, drifts = generate_sequences(TRAIN_DRIFTS, N_TRAIN, seed=seed)
    s_prev, s_t, a_t, s_next = _with_prev(trans)
    T = trans.shape[1]
    sp, st, at, y = (s_prev.reshape(-1, 2), s_t.reshape(-1, 2),
                     a_t.reshape(-1, 2), s_next.reshape(-1, 2))
    ctx = torch.tensor(np.repeat(_drift_context(drifts, condition), T, axis=0), dtype=torch.float32)

    cb = ContextCB(ctx.shape[1]).to(device)
    opt = torch.optim.Adam(cb.parameters(), lr=LR)
    n = len(sp)
    print(f"Training [{condition}] baseline (ctx_dim={ctx.shape[1]})…", flush=True)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            pred = cb(sp[idx].to(device), st[idx].to(device),
                      at[idx].to(device), ctx[idx].to(device))
            loss = F.mse_loss(pred, y[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{condition}] epoch {epoch:03d} | train MSE {tot / n:.4f}", flush=True)
    return cb


def sweep_context_cb(cb, condition, device, seed=999):
    cb.eval()
    mse_mean, mse_sem = [], []
    with torch.no_grad():
        for d in TEST_DRIFTS:
            trans, _ = generate_sequences([d], N_TEST_PER_DRIFT, seed=seed + int(d * 100))
            trans = trans.to(device)
            s_prev, s_t, a_t, s_next = _with_prev(trans)
            ctx_vec = _drift_context(np.full(len(trans), d), condition)
            ctx = (torch.tensor(ctx_vec, dtype=torch.float32, device=device)
                   .unsqueeze(1).expand(-1, trans.shape[1], -1))
            pred = cb(s_prev, s_t, a_t, ctx)
            se = (pred[:, WARMUP:] - s_next[:, WARMUP:]) ** 2
            per_seq = se.mean(dim=(1, 2)).cpu().numpy()
            mse_mean.append(per_seq.mean()); mse_sem.append(per_seq.std() / np.sqrt(len(per_seq)))
    return np.array(mse_mean), np.array(mse_sem)


def plot_overlay(ec_mse, ec_sem, onehot_mse, scalar_mse, pred_drift):
    if _HAS_PLOT_HELPERS:
        set_plt_params(style="fast")
    os.makedirs(FIG_DIR, exist_ok=True)
    d = TEST_DRIFTS
    is_train = np.isin(np.round(d, 3), np.round(TRAIN_DRIFTS, 3))
    held = ~is_train

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.axvspan(min(TRAIN_DRIFTS), max(TRAIN_DRIFTS), color="gray", alpha=0.07, label="trained range")
    for v in TRAIN_DRIFTS:
        ax.axvline(v, color="gray", ls="--", lw=0.8)
    ax.plot(d, onehot_mse, color="tomato",   lw=1.6, marker="o", ms=3, label="one-hot (no structure)")
    ax.plot(d, scalar_mse, color="seagreen", lw=1.6, label="scalar-d oracle (ceiling)")
    ax.fill_between(d, ec_mse - ec_sem, ec_mse + ec_sem, color="steelblue", alpha=0.25)
    ax.plot(d, ec_mse, color="steelblue", lw=2.0, label="EC (inferred, learned structure)")
    ax.set_xlabel("Drift magnitude d")
    ax.set_ylabel(f"CB prediction MSE (steps ≥ {WARMUP})")
    ax.set_title("Structure enables generalization (drift)")
    ax.legend(fontsize=7)
    make_axis_nice(ax)

    if pred_drift is not None:
        ax = axes[1]
        ax.plot([d.min(), d.max()], [d.min(), d.max()], color="gray", ls=":", lw=1, label="identity")
        ax.scatter(d[held], pred_drift[held], color="steelblue", s=18, label="held-out")
        ax.scatter(d[is_train], pred_drift[is_train], color="tomato", s=40, marker="D", label="trained")
        ax.set_xlabel("True drift d"); ax.set_ylabel("Decoded d from z_ec")
        ax.set_title("z_ec encodes drift\n(decoder fit on trained drifts only)")
        ax.legend(fontsize=8)
        make_axis_nice(ax)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"generalization_drift_baselines_{_tag()}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Wrote {out}", flush=True)

    held_in = held & (d >= min(TRAIN_DRIFTS)) & (d <= max(TRAIN_DRIFTS))
    print("\n── Held-out (within-range) mean MSE ──")
    print(f"  EC            : {ec_mse[held_in].mean():.4f}")
    print(f"  scalar-d      : {scalar_mse[held_in].mean():.4f}")
    print(f"  one-hot       : {onehot_mse[held_in].mean():.4f}")


class OracleMLP(nn.Module):
    """[s_{t-1}, a_{t-1}, s_t, d_norm] → 64-D context code (matched to z_ec)."""
    def __init__(self, latent_dim=LATENT_DIM, hidden=EC_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim), nn.Tanh(),
        )

    def forward(self, s_prev, a_prev, s_t, d_norm):
        return self.net(torch.cat([s_prev, a_prev, s_t, d_norm], dim=-1))


def _motion_and_drift(trans, drifts):
    s_t    = trans[:, :, :2]
    a_t    = trans[:, :, 2:4]
    s_next = trans[:, :, 4:6]
    s_prev = torch.cat([s_t[:, :1], s_t[:, :-1]], dim=1)
    a_prev = torch.cat([torch.zeros_like(a_t[:, :1]), a_t[:, :-1]], dim=1)
    T = trans.shape[1]
    dn = (np.asarray(drifts, dtype=np.float32) / _dnorm())
    dn = torch.tensor(dn, device=trans.device).view(-1, 1, 1).expand(-1, T, 1)
    return s_prev, a_prev, s_t, a_t, s_next, dn


def train_mlp_oracle(device, epochs=40, seed=0):
    torch.manual_seed(seed)
    trans, drifts = generate_sequences(TRAIN_DRIFTS, N_TRAIN, seed=seed)
    s_prev, a_prev, s_t, a_t, s_next, dn = _motion_and_drift(trans, drifts)
    flat = lambda x: x.reshape(-1, x.shape[-1])
    sp, ap, st, at, y, dnf = map(flat, (s_prev, a_prev, s_t, a_t, s_next, dn))

    oracle = OracleMLP().to(device)
    cb     = CerebellumCB().to(device)
    opt = torch.optim.Adam(list(oracle.parameters()) + list(cb.parameters()), lr=LR)
    n = len(sp)
    print("Training [matched-oracle] (told the drift, 64-D code)…", flush=True)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            code = oracle(sp[idx].to(device), ap[idx].to(device),
                          st[idx].to(device), dnf[idx].to(device))
            pred = cb(st[idx].to(device), at[idx].to(device), code)
            loss = F.mse_loss(pred, y[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [matched-oracle] epoch {epoch:03d} | train MSE {tot / n:.4f}", flush=True)
    return oracle, cb


def sweep_mlp_oracle(oracle, cb, device, seed=999):
    oracle.eval(); cb.eval()
    mse_mean, mse_sem = [], []
    with torch.no_grad():
        for d in TEST_DRIFTS:
            trans, _ = generate_sequences([d], N_TEST_PER_DRIFT, seed=seed + int(d * 100))
            trans = trans.to(device)
            s_prev, a_prev, s_t, a_t, s_next, dn = _motion_and_drift(trans, np.full(len(trans), d))
            code = oracle(s_prev, a_prev, s_t, dn)
            pred = cb(s_t, a_t, code)
            se = (pred[:, WARMUP:] - s_next[:, WARMUP:]) ** 2
            per_seq = se.mean(dim=(1, 2)).cpu().numpy()
            mse_mean.append(per_seq.mean()); mse_sem.append(per_seq.std() / np.sqrt(len(per_seq)))
    return np.array(mse_mean), np.array(mse_sem)


def plot_matched_oracle(ec_mse, ec_sem, or_mse, or_sem):
    if _HAS_PLOT_HELPERS:
        set_plt_params(style="fast")
    os.makedirs(FIG_DIR, exist_ok=True)
    d = TEST_DRIFTS
    is_train = np.isin(np.round(d, 3), np.round(TRAIN_DRIFTS, 3))
    held = ~is_train
    held_in = held & (d >= min(TRAIN_DRIFTS)) & (d <= max(TRAIN_DRIFTS))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.axvspan(min(TRAIN_DRIFTS), max(TRAIN_DRIFTS), color="gray", alpha=0.07, label="trained range")
    for v in TRAIN_DRIFTS:
        ax.axvline(v, color="gray", ls="--", lw=0.8)
    ax.fill_between(d, or_mse - or_sem, or_mse + or_sem, color="purple", alpha=0.2)
    ax.plot(d, or_mse, color="purple", lw=1.8, label="matched oracle (told d, 64-D)")
    ax.fill_between(d, ec_mse - ec_sem, ec_mse + ec_sem, color="steelblue", alpha=0.25)
    ax.plot(d, ec_mse, color="steelblue", lw=2.0, label="EC (infers d, 64-D)")
    ax.set_xlabel("Drift magnitude d")
    ax.set_ylabel(f"CB prediction MSE (steps ≥ {WARMUP})")
    ax.set_title("EC vs matched oracle (drift)\n(capacity + inputs equalized; only inferred-vs-told differs)")
    ax.legend(fontsize=8)
    make_axis_nice(ax)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"generalization_drift_matched_oracle_{_tag()}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Wrote {out}", flush=True)

    print("\n── Held-out (within-range) mean MSE ──")
    print(f"  EC (infers d) : {ec_mse[held_in].mean():.4f}")
    print(f"  matched oracle: {or_mse[held_in].mean():.4f}")


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot(mse_mean, mse_sem, pred_drift):
    if _HAS_PLOT_HELPERS:
        set_plt_params(style="fast")
    os.makedirs(FIG_DIR, exist_ok=True)
    d = TEST_DRIFTS
    in_range = (d >= min(TRAIN_DRIFTS)) & (d <= max(TRAIN_DRIFTS))
    is_train = np.isin(np.round(d, 3), np.round(TRAIN_DRIFTS, 3))
    held = ~is_train

    n_panels = 2 if pred_drift is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: generalization curve — CB MSE vs drift magnitude
    ax = axes[0]
    ax.fill_between(d, mse_mean - mse_sem, mse_mean + mse_sem, color="seagreen", alpha=0.25)
    ax.plot(d, mse_mean, color="seagreen", lw=1.8, label="CB MSE")
    for v in TRAIN_DRIFTS:
        ax.axvline(v, color="tomato", ls="--", lw=1)
    ax.axvline(TRAIN_DRIFTS[0], color="tomato", ls="--", lw=1, label="trained drifts")
    ax.axvspan(min(TRAIN_DRIFTS), max(TRAIN_DRIFTS), color="gray", alpha=0.07,
               label="trained range")
    ax.set_xlabel("Drift magnitude d")
    ax.set_ylabel(f"CB prediction MSE (steps ≥ {WARMUP})")
    ax.set_title("Generalization across the drift family")
    ax.legend(fontsize=8)
    make_axis_nice(ax)

    # Panel 2: d decoded from z_ec (decoder fit only on trained magnitudes)
    if pred_drift is not None:
        ax = axes[1]
        ax.plot([d.min(), d.max()], [d.min(), d.max()], color="gray", ls=":", lw=1,
                label="identity")
        ax.scatter(d[held], pred_drift[held], color="seagreen", s=18, label="held-out")
        ax.scatter(d[is_train], pred_drift[is_train], color="tomato", s=40, marker="D",
                   label="trained")
        ax.set_xlabel("True drift d")
        ax.set_ylabel("Decoded d from z_ec")
        ax.set_title("z_ec encodes drift as a continuous axis\n(decoder fit on trained drifts only)")
        ax.legend(fontsize=8)
        make_axis_nice(ax)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"generalization_drift_{_tag()}")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Wrote {out}", flush=True)

    interp = held & in_range
    extrap = held & ~in_range
    print("\n── Held-out MSE summary ──")
    print(f"  interpolation (within trained range) : {mse_mean[interp].mean():.4f}")
    if extrap.any():
        print(f"  extrapolation (outside trained range): {mse_mean[extrap].mean():.4f}")
    print(f"  trained drifts                        : {mse_mean[is_train].mean():.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-train", action="store_true",
                   help="Load existing checkpoint instead of retraining.")
    p.add_argument("--baselines", action="store_true",
                   help="Also train one-hot + scalar-d baselines and make the overlay figure.")
    p.add_argument("--matched-oracle", action="store_true",
                   help="Also train the matched oracle (told d, same inputs, 64-D code).")
    p.add_argument("--train-drifts", type=float, nargs="+", default=None,
                   help="Override training drift magnitudes, e.g. --train-drifts 0.4 1.2 to "
                        "force a spike at 0.8. Checkpoints/figures are tagged by the drift set.")
    args = p.parse_args()

    global TRAIN_DRIFTS
    if args.train_drifts is not None:
        TRAIN_DRIFTS = list(args.train_drifts)
        print(f"Training drifts overridden to: {TRAIN_DRIFTS}", flush=True)

    t0 = time.time()
    if args.skip_train and os.path.exists(ckpt_path()):
        ec, cb, device = load()
        print(f"Loaded {ckpt_path()}", flush=True)
    else:
        ec, cb, device = train(seed=args.seed)

    mse_mean, mse_sem, z_by_drift = sweep(ec, cb, device)
    pred_drift = drift_decode(z_by_drift)
    plot(mse_mean, mse_sem, pred_drift)

    if args.baselines:
        onehot = train_context_cb("onehot", device, seed=args.seed)
        oh_mse, _ = sweep_context_cb(onehot, "onehot", device)
        scalar = train_context_cb("scalar", device, seed=args.seed)
        sc_mse, _ = sweep_context_cb(scalar, "scalar", device)
        plot_overlay(mse_mean, mse_sem, oh_mse, sc_mse, pred_drift)

    if args.matched_oracle:
        oracle, ocb = train_mlp_oracle(device, seed=args.seed)
        or_mse, or_sem = sweep_mlp_oracle(oracle, ocb, device)
        plot_matched_oracle(mse_mean, mse_sem, or_mse, or_sem)

    print(f"\nDone in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

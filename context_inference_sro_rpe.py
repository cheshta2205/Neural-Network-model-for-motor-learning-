"""
context_inference_sro_rpe.py

Can EC infer the context `c` when the error signal coming from CB isn't state pred error but REWARD-prediction error, RPE
Task — Bernardi et al. 2020 Cell serial-reversal SRO 

  • 4 stimuli per context (one-hot), 2 latent contexts, never cued.
  • Each (stimulus, context) defines an SRO mapping:
        operant O*  ∈ {hold=0, release=1}   — the correct behavioural response
        reward  r*  ∈ {no=0,  reward=1}      — reinforcement for correct perf.
    Operant and reinforcement are ORTHOGONAL: the 4 stimuli in a context span
    {R+, R−, H+, H−}.
  • Context switches every 50–70 trials; ALL 4 mappings flip at once; many
    switches per sequence. agent needs to infer the context im in and which stim response will get me reward

Model — MTL→CB closed loop (architecture of NN same as ECtoCB_train_MSE_12c.py), task changes and pred error changes. 

  EC (GRU)  : z_t = EC([x_{t-1}, x_t, δ_{t-1}])          --  signal is the RPE δ
  CB (MLP)  : (Ô_t, r̂_t) = CB(x_t, z_t)                  -- operant + reward heads
  RPE       : δ_t = r*_t − r̂_t   (scalar, fed back to EC; replaces s_t − s_pred)
  Loss      : BCE(Ô, O*) + MSE(r̂, r*)

  plot - 

  1. Decode c from z_ec  (chance = 0.5).               -- can MTL infer context?
  2. Context decode accuracy vs trials-since-switch.   →  can z_ec decode a after a switch?
  3. Fig-1C inference: operant accuracy on the FIRST post-switch appearance of each stimulus, vs its appearance-order (1–4). Image 1 uses the stale
context (≤ chance); images 2–4 above chance = inference from partial evidence.
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

try:
    from src.PlottingTools.plotting_functions import set_plt_params, make_axis_nice
    _HAS_PLOT_HELPERS = True
except Exception:                                    # pragma: no cover
    _HAS_PLOT_HELPERS = False
    def set_plt_params(*a, **k): pass
    def make_axis_nice(*a, **k): pass


# Config
N_STIM   = 4
N_CTX    = 2
SEQ_LEN  = 240
SWITCH_MIN, SWITCH_MAX = 50, 70      # trials per block (uncued switch)

N_TRAIN  = 2000
N_VAL    = 300
BATCH    = 128
EPOCHS   = 30
LR       = 1e-3

LATENT_DIM = 32
EC_HIDDEN  = 256
CB_HIDDEN  = 256
EC_INPUT   = 2 * N_STIM + 1           # [x_{t-1}(4), x_t(4), δ_{t-1}(1)] = 9

CKPT = "checkpoints/context_inference_sro_rpe.pt"
FIG_DIR = "paper/context_inference_sro"

# ── SRO table: SRO[context, stimulus] = (operant, reward) 
#   context 0:  stim0=R+  stim1=R−  stim2=H+  stim3=H−   (spans the 2×2)
#   context 1:  full reversal — every operant AND reward flips, mappings change
#               for all 4 stimuli, while the within-context 2×2 is preserved.
#   reward identifies context (ctx0 rewards {0,2}; ctx1 rewards {1,3}).
SRO = np.array([
    [[1, 1], [1, 0], [0, 1], [0, 0]],   # context 0
    [[0, 0], [0, 1], [1, 0], [1, 1]],   # context 1
], dtype=np.float32)                     # shape (2, 4, 2)


# ── Data: serial-reversal sequences 
def generate_sequences(n_seqs, seq_len=SEQ_LEN, seed=0):
    """Returns stim (N,T int), ctx (N,T int), O* (N,T), r* (N,T)."""
    rng = np.random.default_rng(seed)
    stim = np.zeros((n_seqs, seq_len), dtype=np.int64)
    ctx  = np.zeros((n_seqs, seq_len), dtype=np.int64)
    Ostar = np.zeros((n_seqs, seq_len), dtype=np.float32)
    rstar = np.zeros((n_seqs, seq_len), dtype=np.float32)

    for n in range(n_seqs):
        # build the (uncued) context schedule
        c = int(rng.integers(0, N_CTX))
        t = 0
        while t < seq_len:
            blk = int(rng.integers(SWITCH_MIN, SWITCH_MAX + 1))
            ctx[n, t:t + blk] = c
            t += blk
            c = 1 - c                                   # flip context at block end
        # random stimulus each trial → look up SRO mapping
        s = rng.integers(0, N_STIM, size=seq_len)
        stim[n] = s
        Ostar[n] = SRO[ctx[n], s, 0]
        rstar[n] = SRO[ctx[n], s, 1]

    return (torch.tensor(stim), torch.tensor(ctx),
            torch.tensor(Ostar), torch.tensor(rstar))


# ── Models
class EntorhinalCortex(nn.Module):
    """GRU-cell EC. Signal carried in its input is the scalar RPE δ_{t-1}."""
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
    """MLP: [stim_onehot, z] → (operant logit, reward). Replaces s_{t+1} head."""
    def __init__(self, n_stim=N_STIM, latent_dim=LATENT_DIM, hidden=CB_HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_stim + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),              nn.ReLU(),
        )
        self.operant_head = nn.Linear(hidden, 1)     # hold/release logit
        self.reward_head  = nn.Linear(hidden, 1)     # expected reward

    def forward(self, x_onehot, z):
        h = self.trunk(torch.cat([x_onehot, z], dim=-1))
        return self.operant_head(h), self.reward_head(h)


# ── Closed-loop rollout: CB's RPE is fed back into EC 
def run_sequence(ec, cb, stim, ctx, Ostar, rstar, device):
    """Step through the batch, feeding the scalar RPE δ_t = r* − r̂ back into EC.

    Causal: z_t is built from past stimuli + δ_{t-1} only, so it cannot peek at
    the current reward. δ is detached when fed to EC — it is a feedback *signal*,
    not a gradient path (learning is driven by the prediction losses).
    """
    B, T = stim.shape
    x_oh = F.one_hot(stim, N_STIM).float().to(device)           # (B,T,4)
    rstar = rstar.to(device)

    h        = ec.init_hidden(B, device)
    x_prev   = torch.zeros(B, N_STIM, device=device)
    delta_pr = torch.zeros(B, 1, device=device)

    z_seq, Olog_seq, rpred_seq = [], [], []
    for t in range(T):
        x_t = x_oh[:, t]
        ec_in   = torch.cat([x_prev, x_t, delta_pr], dim=-1)    # (B, 9)
        z_t, h  = ec.step(ec_in, h)
        O_logit, r_pred = cb(x_t, z_t)                          # (B,1),(B,1)

        delta_pr = (rstar[:, t:t + 1] - r_pred).detach()        # RPE → next step
        x_prev   = x_t

        z_seq.append(z_t); Olog_seq.append(O_logit); rpred_seq.append(r_pred)

    return (torch.stack(z_seq, 1),                              # (B,T,latent)
            torch.stack(Olog_seq, 1).squeeze(-1),              # (B,T)
            torch.stack(rpred_seq, 1).squeeze(-1))             # (B,T)


# ── Train 
def train(seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | SRO serial-reversal | RPE signal to EC", flush=True)

    stim, ctx, Ostar, rstar = generate_sequences(N_TRAIN, seed=seed)
    ec = EntorhinalCortex().to(device)
    cb = CerebellumCB().to(device)
    opt = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=LR)

    n = N_TRAIN
    for epoch in range(1, EPOCHS + 1):
        ec.train(); cb.train()
        perm = torch.randperm(n)
        tot_o = tot_r = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            z, Olog, rpred = run_sequence(ec, cb, stim[idx], ctx[idx],
                                          Ostar[idx], rstar[idx], device)
            O_t = Ostar[idx].to(device)
            r_t = rstar[idx].to(device)
            loss_o = F.binary_cross_entropy_with_logits(Olog, O_t)
            loss_r = F.mse_loss(rpred, r_t)
            loss = loss_o + loss_r
            opt.zero_grad(); loss.backward(); opt.step()
            tot_o += loss_o.item() * len(idx); tot_r += loss_r.item() * len(idx)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:03d} | operant BCE {tot_o/n:.4f} | reward MSE {tot_r/n:.4f}",
                  flush=True)

    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "seed": seed}, CKPT)
    print(f"Saved {CKPT}", flush=True)
    return ec, cb, device


def load(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, map_location=device)
    ec = EntorhinalCortex().to(device); ec.load_state_dict(ck["ec"]); ec.eval()
    cb = CerebellumCB().to(device);     cb.load_state_dict(ck["cb"]); cb.eval()
    return ec, cb, device


# ── Analyses
def gather(ec, cb, device, seed=999):
    ec.eval(); cb.eval()
    stim, ctx, Ostar, rstar = generate_sequences(N_VAL, seed=seed)
    with torch.no_grad():
        z, Olog, rpred = run_sequence(ec, cb, stim, ctx, Ostar, rstar, device)
    return (stim.numpy(), ctx.numpy(), Ostar.numpy(), rstar.numpy(),
            z.cpu().numpy(), Olog.cpu().numpy(), rpred.cpu().numpy())


def trials_since_switch(ctx):
    """(N,T) int: 0 on a switch trial (or t=0), incrementing within a block."""
    N, T = ctx.shape
    tss = np.zeros((N, T), dtype=int)
    for n in range(N):
        k = 0
        for t in range(1, T):
            k = 0 if ctx[n, t] != ctx[n, t - 1] else k + 1
            tss[n, t] = k
    return tss


def decode_context(z, ctx):
    """5-fold logistic decode of context from z_ec (chance = 1/N_CTX)."""
    Z = z.reshape(-1, z.shape[-1]); C = ctx.reshape(-1)
    return cross_val_score(LogisticRegression(max_iter=1000), Z, C,
                           cv=5, scoring="accuracy").mean()


def decode_vs_tss(z, ctx, max_tss=40):
    """Fit one decoder on all (z,ctx); report accuracy vs trials-since-switch."""
    Z = z.reshape(-1, z.shape[-1]); C = ctx.reshape(-1)
    clf = LogisticRegression(max_iter=1000).fit(Z, C)
    pred = clf.predict(Z).reshape(ctx.shape)
    tss = trials_since_switch(ctx)
    acc = []
    for k in range(max_tss):
        m = tss == k
        acc.append((pred[m] == ctx[m]).mean() if m.any() else np.nan)
    return np.array(acc)


def decode_around_switch(z, ctx, pre=5, post=20):
    """Fit one decoder on all (z,ctx); accuracy vs SIGNED offset from a switch.
    lag < 0 = pre-switch (still old block), lag 0 = first trial of new block,
    lag > 0 = post-switch. Shows the baseline collapse and re-locking together."""
    Z = z.reshape(-1, z.shape[-1]); C = ctx.reshape(-1)
    clf = LogisticRegression(max_iter=1000).fit(Z, C)
    pred = clf.predict(Z).reshape(ctx.shape)
    N, T = ctx.shape
    lags = np.arange(-pre, post + 1)
    buckets = {int(L): [] for L in lags}
    for n in range(N):
        switches = [t for t in range(1, T) if ctx[n, t] != ctx[n, t - 1]]
        for ts in switches:
            for L in lags:
                t = ts + int(L)
                if 0 <= t < T:
                    buckets[int(L)].append(pred[n, t] == ctx[n, t])
    acc = np.array([np.mean(buckets[int(L)]) if buckets[int(L)] else np.nan
                    for L in lags])
    return lags, acc


def inference_curve(stim, ctx, Olog, Ostar):
    """Fig-1C analog: operant accuracy on the FIRST post-switch appearance of
    each stimulus, bucketed by appearance order (1=first stim seen after a
    switch, …). Order 1 uses the stale context; orders 2–4 test inference."""
    N, T = stim.shape
    O_pred = (Olog > 0).astype(int)            # sigmoid(logit) > 0.5
    correct = (O_pred == Ostar.astype(int))
    buckets = {k: [] for k in range(1, N_STIM + 1)}
    for n in range(N):
        seen = set()
        for t in range(T):
            if t == 0 or ctx[n, t] != ctx[n, t - 1]:
                seen = set()                    # reset at switch / sequence start
            s = stim[n, t]
            if s not in seen:
                order = len(seen) + 1
                if order <= N_STIM:
                    buckets[order].append(correct[n, t])
                seen.add(s)
    mean = {k: (np.mean(v) if v else np.nan) for k, v in buckets.items()}
    nobs = {k: len(v) for k, v in buckets.items()}
    return mean, nobs


def evaluate_and_plot(ec, cb, device):
    stim, ctx, Ostar, rstar, z, Olog, rpred = gather(ec, cb, device)

    acc_c        = decode_context(z, ctx)
    lags, acc_lag = decode_around_switch(z, ctx)
    infer, nobs = inference_curve(stim, ctx, Olog, Ostar)
    overall_op = ((Olog > 0).astype(int) == Ostar.astype(int)).mean()

    print("\n── Results ──")
    print(f"  Decode context c from z_ec : {acc_c:.3f}   (chance = {1/N_CTX:.2f})")
    print(f"  Overall operant accuracy   : {overall_op:.3f}")
    print(f"  Fig-1C inference (operant accuracy by post-switch appearance order):")
    for k in range(1, N_STIM + 1):
        print(f"     image {k}: {infer[k]:.3f}   (n={nobs[k]})")

    if _HAS_PLOT_HELPERS:
        set_plt_params(style="fast")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: context decode collapses at the switch, then re-locks
    ax = axes[0]
    ax.axhline(1 / N_CTX, color="gray", ls="--", lw=1, label="chance")
    ax.axvline(0, color="tomato", ls=":", lw=1.2, label="switch")
    ax.plot(lags, acc_lag, color="steelblue", lw=2, marker="o", ms=3)
    ax.set_xlabel("Trials relative to context switch")
    ax.set_ylabel("Context decode accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"EC infers c from a SCALAR RPE\n(overall decode = {acc_c:.2f})")
    ax.legend(fontsize=8); make_axis_nice(ax)

    # Panel 2: Fig-1C inference signature
    ax = axes[1]
    ks = list(range(1, N_STIM + 1))
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    ax.plot(ks, [infer[k] for k in ks], color="tomato", lw=2, marker="s", ms=6)
    ax.set_xticks(ks)
    ax.set_xlabel("Stimulus appearance order after switch")
    ax.set_ylabel("Operant accuracy (first appearance)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Inference signature (cf. Bernardi Fig 1C)\n"
                 "order 1 = stale context; 2–4 = inferred")
    ax.legend(fontsize=8); make_axis_nice(ax)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "context_inference_sro_rpe.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {out}", flush=True)


# ── Main 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    global EPOCHS
    if args.epochs is not None:
        EPOCHS = args.epochs

    t0 = time.time()
    if args.skip_train and os.path.exists(CKPT):
        ec, cb, device = load()
        print(f"Loaded {CKPT}", flush=True)
    else:
        ec, cb, device = train(seed=args.seed)

    evaluate_and_plot(ec, cb, device)
    print(f"\nDone in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

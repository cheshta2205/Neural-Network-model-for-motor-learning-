"""
pretrain_thalamus.py

Pretrain the Thalamus router using cross-entropy with context labels.

Pipeline:
    1. Load frozen EC from ectocb_8ctx_seq80_z32.pt  (MSE-only trained)
    2. Generate sequences switching between CTX_A (rot+60) and CTX_B (rot0)
    3. Run frozen EC forward → z_ec at each timestep (B, T, 32)
    4. Train thalamus.router with cross-entropy:
           loss = CrossEntropy(router(z_ec_t), context_id_t)
    5. Save pretrained thalamus to checkpoints/thalamus_pretrained.pt

Router weights will then be FROZEN during MC training (MC_savings_Thalamus_.py)
so routing cannot collapse from task loss gradients.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ── Hyperparameters ────────────────────────────────────────────────────────────
LATENT_DIM  = 32
EC_HIDDEN   = 256
CB_HIDDEN   = 256
EC_INPUT    = 8
N_EXPERTS   = 2       # CTX_A=0, CTX_B=1
SEQ_LEN     = 200  # long sequences — EC state persists, z_smooth converges
WARMUP      = 50   # skip first N timesteps per sequence (z_smooth still building up)
GRID_SIZE   = 4.0
DT          = 1.0

N_TRAIN     = 5000
N_VAL       = 500
BATCH_SIZE  = 128
EPOCHS      = 30
LR          = 1e-3
EMA_ALPHA   = 0.5    # must match MC_savings_Thalamus_.py


CKPT_EC     = "checkpoints/ectocb_8ctx_seq80_z32.pt"
CKPT_OUT    = "checkpoints/thalamus_pretrained_rot45curl_persistent_ema05.pt"
FIG_DIR     = "../../figs/ECtoCB/thalamus_pretrain_rot45curl_persistent_ema05"

# ── 2 ABA contexts — same as MC_savings_Thalamus_.py ─────────────────────────
# c_id: 0=CTX_A (rot+45), 1=CTX_B (curl+)
CONTEXTS = [
    (1.0,  np.pi/4,  0.0,  0.0,  0.0,  0.0,  0,  "A: rot+45"),
    (1.0,  0.0,      1.0,  0.0,  0.0,  0.0,  1,  "B: curl+"),
]


# ── Environment step ───────────────────────────────────────────────────────────
def env_step(pos, vel, action, beta, theta, gamma, g, bx, by):
    R    = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
    curl = np.array([[0., -1.], [1., 0.]])
    b    = np.array([bx, by], dtype=np.float32)
    F    = beta * (R @ action) + gamma * (curl @ vel) + g * b
    vel_new = vel + F
    pos_new = np.clip(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new.astype(np.float32), vel_new.astype(np.float32)


# ── Data generation ────────────────────────────────────────────────────────────
def generate_sequences(n_seqs, seq_len=SEQ_LEN, seed=0):
    """
    Same as ECtoCB_newcontexts_train.py.
    Each sequence: 2 contexts chosen without replacement, switch at t~U[40,60].

    Returns:
        transitions : (N, T, 6)   [s_{t-1}(2), a_{t-1}(2), s_t(2)]
        ctx_labels  : (N, T, 3)   [beta, gamma, c_id]
    """
    rng = np.random.default_rng(seed)
    all_trans, all_ctx = [], []

    for _ in range(n_seqs):
        pos    = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
        vel    = np.zeros(2, dtype=np.float32)
        # Single context per sequence — EC state persists, matches MC training
        ctx    = CONTEXTS[rng.integers(len(CONTEXTS))]

        seq_trans, seq_ctx = [], []

        for t in range(seq_len):
            beta, theta, gamma, g, bx, by, c_id, _ = ctx
            action = rng.uniform(-1, 1, size=2).astype(np.float32)
            pos_next, vel_next = env_step(pos, vel, action, beta, theta, gamma, g, bx, by)
            seq_trans.append([pos[0], pos[1], action[0], action[1],
                               pos_next[0], pos_next[1]])
            seq_ctx.append([beta, gamma, float(c_id)])
            pos = pos_next
            vel = vel_next

        all_trans.append(seq_trans)
        all_ctx.append(seq_ctx)

    return (
        torch.tensor(np.array(all_trans), dtype=torch.float32),  # (N, T, 6)
        torch.tensor(np.array(all_ctx),   dtype=torch.float32),  # (N, T, 3)
    )


# ── EC (identical to ECtoCB_newcontexts_train.py) ─────────────────────────────
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


# ── CB (identical to MC_savings_Thalamus_.py) ─────────────────────────────────
class CerebellumCB(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(4 + LATENT_DIM, CB_HIDDEN), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(CB_HIDDEN, CB_HIDDEN),       nn.ReLU())
        self.out    = nn.Linear(CB_HIDDEN, 2)

    def forward(self, s, a, z):
        h = self.layer1(torch.cat([s, a, z], dim=-1))
        return self.out(self.layer2(h))


# ── Thalamus router ───────────────────────────────────────────────────────────
class Thalamus(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, n_experts=N_EXPERTS):
        super().__init__()
        self.router = nn.Linear(latent_dim, n_experts)

    def forward(self, z_ec):
        return self.router(z_ec)   # raw logits (B, n_experts)


# ── Get z_ec for a batch of sequences ─────────────────────────────────────────
@torch.no_grad()
def get_zec(ec, cb, transitions, device):
    """
    Run frozen EC+CB step-by-step over sequences.
    EC input at t: [s_{t-1}(2), a_{t-1}(2), s_t(2), e_t(2)]
    e_t = s_{t+1} - CB(s_t, a_t, z_{t-1})  — real CB prediction error.

    Matches what MC_savings_Thalamus_.py does during training so the
    router trains on the same z_ec distribution it will see at eval time.

    Returns z_smooth: (B, T, LATENT_DIM)
    """
    B, T, _ = transitions.shape
    h        = ec.init_hidden(B, device)
    zeros2   = torch.zeros(B, 2, device=device)
    z_smooth = torch.zeros(B, LATENT_DIM, device=device)
    z_t      = torch.zeros(B, LATENT_DIM, device=device)  # z_ec from previous step (fed to CB)
    z_seq    = []

    for t in range(T):
        s_t    = transitions[:, t, :2]
        a_t    = transitions[:, t, 2:4]
        s_next = transitions[:, t, 4:6]
        s_tm1  = transitions[:, t - 1, :2]  if t > 0 else zeros2
        a_tm1  = transitions[:, t - 1, 2:4] if t > 0 else zeros2

        # CB prediction error using z_t from previous step — matches MC training
        e_t = s_next - cb(s_t, a_t, z_t)   # (B, 2)

        ec_input = torch.cat([s_tm1, a_tm1, s_t, e_t], dim=-1)  # (B, 8)
        z_t, h   = ec.step(ec_input, h)
        z_smooth = EMA_ALPHA * z_smooth + (1 - EMA_ALPHA) * z_t
        z_seq.append(z_smooth)

    return torch.stack(z_seq, dim=1)   # (B, T, LATENT_DIM)


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load frozen EC + CB
    ckpt = torch.load(CKPT_EC, map_location=device)
    ec   = EntorhinalCortex().to(device)
    ec.load_state_dict(ckpt["ec"])
    ec.eval()
    for p in ec.parameters():
        p.requires_grad_(False)

    cb     = CerebellumCB().to(device)
    cb_sd  = ckpt["cb"]
    remap  = {"net.0": "layer1.0", "net.2": "layer2.0", "net.4": "out"}
    new_sd = {}
    for k, v in cb_sd.items():
        prefix = ".".join(k.split(".")[:2])
        new_sd[k.replace(prefix, remap[prefix]) if prefix in remap else k] = v
    cb.load_state_dict(new_sd)
    cb.eval()
    for p in cb.parameters():
        p.requires_grad_(False)
    print(f"Loaded frozen EC + CB from {CKPT_EC}")

    # Generate data
    print("Generating sequences...")
    tr_trans, tr_ctx = generate_sequences(N_TRAIN, seed=0)
    va_trans, va_ctx = generate_sequences(N_VAL,   seed=99)

    # Pre-compute z_ec for all sequences (EC is frozen — only do this once)
    print("Computing z_ec for all sequences...")
    tr_zec = []
    for i in range(0, N_TRAIN, BATCH_SIZE):
        batch = tr_trans[i:i+BATCH_SIZE].to(device)
        tr_zec.append(get_zec(ec, cb, batch, device).cpu())
    tr_zec = torch.cat(tr_zec)   # (N_TRAIN, T, LATENT_DIM)

    va_zec = []
    for i in range(0, N_VAL, BATCH_SIZE):
        batch = va_trans[i:i+BATCH_SIZE].to(device)
        va_zec.append(get_zec(ec, cb, batch, device).cpu())
    va_zec = torch.cat(va_zec)   # (N_VAL, T, LATENT_DIM)

    va_cids = va_ctx[:, :, 2].long()   # (N_VAL, T) — used for routing accuracy eval

    thalamus  = Thalamus().to(device)
    optimizer = torch.optim.Adam(thalamus.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\nPretraining thalamus router with cross-entropy ({EPOCHS} epochs)...\n")

    for epoch in range(1, EPOCHS + 1):
        thalamus.train()
        perm = torch.randperm(N_TRAIN)
        tr_loss_sum = 0.0

        for i in range(0, N_TRAIN, BATCH_SIZE):
            idx   = perm[i:i+BATCH_SIZE]
            z_b   = tr_zec[idx].to(device)              # (B, T, 32)
            c_b   = tr_ctx[idx, :, 2].long().to(device) # (B, T) context ids
            B, T, _ = z_b.shape

            # Skip warmup timesteps — only train on converged z_smooth
            z_b_post = z_b[:, WARMUP:, :]               # (B, T-WARMUP, 32)
            c_b_post = c_b[:, WARMUP:]                   # (B, T-WARMUP)
            B, T2, _ = z_b_post.shape
            logits = thalamus(z_b_post.reshape(B * T2, -1))
            labels = c_b_post.reshape(B * T2)
            loss   = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item() * B

        train_loss = tr_loss_sum / N_TRAIN

        thalamus.eval()
        with torch.no_grad():
            # Val CE loss
            va_loss_sum = 0.0
            for i in range(0, N_VAL, BATCH_SIZE):
                z_b = va_zec[i:i+BATCH_SIZE, WARMUP:, :].to(device)
                c_b = va_ctx[i:i+BATCH_SIZE, WARMUP:, 2].long().to(device)
                B, T2, _ = z_b.shape
                logits = thalamus(z_b.reshape(B * T2, -1))
                labels = c_b.reshape(B * T2)
                va_loss_sum += F.cross_entropy(logits, labels).item() * B
            val_loss = va_loss_sum / N_VAL

            # Routing accuracy
            va_zec_flat  = va_zec.reshape(-1, LATENT_DIM).to(device)
            va_cids_flat = va_cids.reshape(-1).to(device)
            val_acc = (thalamus(va_zec_flat).argmax(dim=-1) == va_cids_flat).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d} | "
              f"Train CE: {train_loss:.4f} | "
              f"Val CE: {val_loss:.4f}  routing_acc: {val_acc:.4f}")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "thalamus_state_dict": thalamus.state_dict(),
        "n_experts": N_EXPERTS,
        "latent_dim": LATENT_DIM,
    }, CKPT_OUT)
    print(f"\nSaved pretrained thalamus to {CKPT_OUT}")

    # Plot
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(history["val_acc"], label="val (context labels)")
    axes[1].set_title("Routing accuracy (eval only, not used in loss)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(1/N_EXPERTS, color="gray", ls="--", lw=1, label=f"chance ({1/N_EXPERTS:.2f})")
    axes[1].legend()
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Thalamus router pretraining\n"
                 "Cross-entropy (context labels) | EC+CB frozen | rot+45 vs curl+ | real CB errors")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/thalamus_pretrain.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {FIG_DIR}/thalamus_pretrain.png")


if __name__ == "__main__":
    train()

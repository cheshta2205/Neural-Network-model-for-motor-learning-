"""
train_ec_hc_cb.py

Full pipeline: EC → HC → CB trained end-to-end.

Context is NEVER explicitly given to the model. EC must infer context 
from sequences of transitions [s_t, a_t, s_{t+1}].

We use temporal proximity as a method to cluster transitions such that if
transitions are closer to each other in time (Δt small), the similarity
between z_t and z_{t+Δt} will be high, and vice versa.


Two losses:
  1. MSE loss:       CB(s_t, a_t, c_t) predicts s_{t+1} accurately
  2. InfoNCE loss:   temporal proximity as proxy for context
                     close in time (small Δt)  → same context → z_t similar   (β)
                     far in time   (large Δt') → diff context → z_t different  (α)
                     Loss = -log( β / (β + α) )  →  β/(β+α) → 1

Data: long sequences (seq_len=40) where context changes every context_duration=20 steps
      2 context blocks per sequence: |--ctx A (t=0-19)--|--ctx B (t=20-39)--|

      Close pairs  (Δt ≤ 3)  : likely same context block  → positive
      Far pairs    (Δt ≥ 21) : likely different context block → negative

Architecture:
  EC  : GRU, input [s_t, a_t, s_{t+1}] (6-dim) → z_t (16-dim)
  HC  : leaky integrator, z_t (16-dim) → c_t (16-dim)
  CB  : MLP, [s_t, a_t, c_t] (2+2+16=20-dim) → ŝ_{t+1} (2-dim)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from ec_module import EntorhinalCortex
from hc_module import Hippocampus


# Combined model of EC, HC and CB 

class EC_HC_CB(nn.Module):
    def __init__(self, ec_hidden=256, latent_dim=16, hc_dim=16, cb_hidden=256):
        super().__init__()
        self.ec = EntorhinalCortex(input_dim=6, hidden_dim=ec_hidden, latent_dim=latent_dim)
        self.hc = Hippocampus(latent_dim=latent_dim, hc_dim=hc_dim)

        # CB takes [s_t (2), a_t (2), c_t (hc_dim)] = 2+2+hc_dim
        cb_input_dim = 2 + 2 + hc_dim
        self.cb = nn.Sequential(
            nn.Linear(cb_input_dim, cb_hidden),
            nn.ReLU(),
            nn.Linear(cb_hidden, cb_hidden),
            nn.ReLU(),
            nn.Linear(cb_hidden, 2),
        )

    def forward(self, transitions, states, actions):
        """
        Args:
            transitions : (B, T, 6)  [s_t, a_t, s_{t+1}] — EC input
            states      : (B, T, 2)  s_t — CB input
            actions     : (B, T, 2)  a_t — CB input

        Returns:
            s_pred : (B, T, 2)  predicted next state
            z      : (B, T, 16) context estimates from EC (for InfoNCE loss)
        """
        z, _    = self.ec(transitions)       # (B, T, 16)
        c, _    = self.hc(z)                 # (B, T, 16)

        # Concatenate s_t, a_t, c_t and predict next state at every timestep
        cb_input = torch.cat([states, actions, c], dim=-1)  # (B, T, 20)
        s_pred   = self.cb(cb_input)                         # (B, T, 2)

        return s_pred, z


# InfoNCE loss (temporal proximity) 
# Anchor:   z_t           at random timestep t
# Positive: z_{t+Δt}      small Δt  (close in time → same context)   → β
# Negative: z_{t+Δt'}     large Δt' (far in time   → diff context)   → α
#
# Loss = -log( β / (β + α) )
#      = cross_entropy( [β/τ, α/τ], label=0 )   ← positive is class 0

def infonce_loss(z, small_delta=3, large_delta=21, temperature=0.1):
    """
    z           : (B, T, D)
    small_delta : max timestep gap for positive pair  (same context likely)
    large_delta : min timestep gap for negative pair  (different context likely)
    """
    B, T, D = z.shape

    # Anchor range: must allow both positive and negative offsets
    # anchor t ∈ [0, T - large_delta - 1] #anchor will only be sampled from this range 
    max_anchor = T - large_delta - 1
    assert max_anchor > 0, f"seq_len {T} too short for large_delta {large_delta}"

    t_anchor = torch.randint(0, max_anchor, (B,), device=z.device)

    # Positive: close in time (Δt ∈ [1, small_delta])
    dt_pos = torch.randint(1, small_delta + 1, (B,), device=z.device)
    t_pos  = t_anchor + dt_pos                            # still within sequence

    # Negative: far in time (Δt' ∈ [large_delta, T - t_anchor - 1])
    dt_neg = torch.randint(large_delta, T - max_anchor, (B,), device=z.device)
    t_neg  = t_anchor + dt_neg

    anchors   = z[torch.arange(B), t_anchor, :]   # (B, D)
    positives = z[torch.arange(B), t_pos,    :]   # (B, D)
    negatives = z[torch.arange(B), t_neg,    :]   # (B, D)

    # Normalise → cosine similarity
    anchors   = F.normalize(anchors,   dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # β = <anchor, positive>,  α = <anchor, negative>
    beta  = (anchors * positives).sum(dim=-1, keepdim=True)  # (B, 1)
    alpha = (anchors * negatives).sum(dim=-1, keepdim=True)  # (B, 1)

    # Stack: [β, α] / τ → cross_entropy with label 0 (positive is first)
    logits = torch.cat([beta, alpha], dim=-1) / temperature  # (B, 2)
    labels = torch.zeros(B, dtype=torch.long, device=z.device)
    loss   = F.cross_entropy(logits, labels)

    return loss


# Sequence dataset
# Each sequence is seq_len=40 timesteps long.
# Context changes every context_duration=20 steps → 2 context blocks per sequence:
#   t=0..19  : context A (beta_A, gamma_A)
#   t=20..39 : context B (beta_B, gamma_B)
#
# EC sees only transitions [s_t, a_t, s_{t+1}] — context is NEVER given.
# Temporal proximity assumption:
#   close pairs (Δt ≤ 3)  → likely same context block → should be similar
#   far pairs   (Δt ≥ 21) → likely different context block → should be different

def generate_sequences(n_seqs, seq_len=40, context_duration=20, seed=0):
    """
    Returns:
      transitions : (N, T, 6)  [s_t, a_t, s_{t+1}]  ← EC input only
      states      : (N, T, 2)  s_t                   ← CB input
      actions     : (N, T, 2)  a_t                   ← CB input
      next_states : (N, T, 2)  s_{t+1}               ← prediction target
      true_contexts:(N, T, 2)  [beta, gamma] at each t  ← evaluation only
    """
    rng  = np.random.default_rng(seed)
    acts = [(-1., 0.), (1., 0.), (0., -1.), (0., 1.)]
    n_blocks = seq_len // context_duration

    all_trans, all_s, all_a, all_ns, all_c = [], [], [], [], []

    for _ in range(n_seqs):
        x, y = float(rng.integers(0, 5)), float(rng.integers(0, 5))

        # Sample a new context for each block within the sequence
        betas  = rng.uniform(-0.5, 0.5, size=n_blocks)
        gammas = rng.uniform(-0.5, 0.5, size=n_blocks)

        seq_trans, seq_s, seq_a, seq_ns, seq_ctx = [], [], [], [], []

        for t in range(seq_len):
            # Which context block are we in?
            block = t // context_duration
            beta, gamma = float(betas[block]), float(gammas[block])

            dx, dy = acts[rng.integers(0, 4)]
            a      = np.array([dx, dy], dtype=np.float32)
            a_norm = np.linalg.norm(a)
            a_hat  = a / a_norm if a_norm > 0 else np.zeros(2, dtype=np.float32)
            a_hat_perp = np.array([-a_hat[1], a_hat[0]], dtype=np.float32)

            f_env  = beta * a_hat + gamma * a_hat_perp
            x_next = float(np.clip(x + dx + f_env[0], 0, 4))
            y_next = float(np.clip(y + dy + f_env[1], 0, 4))

            seq_trans.append([x, y, dx, dy, x_next, y_next])
            seq_s.append([x, y])
            seq_a.append([dx, dy])
            seq_ns.append([x_next, y_next])
            seq_ctx.append([beta, gamma])

            x, y = x_next, y_next

        all_trans.append(seq_trans)
        all_s.append(seq_s)
        all_a.append(seq_a)
        all_ns.append(seq_ns)
        all_c.append(seq_ctx)

    return (
        torch.tensor(np.array(all_trans), dtype=torch.float32),  # (N, T, 6)
        torch.tensor(np.array(all_s),     dtype=torch.float32),  # (N, T, 2)
        torch.tensor(np.array(all_a),     dtype=torch.float32),  # (N, T, 2)
        torch.tensor(np.array(all_ns),    dtype=torch.float32),  # (N, T, 2)
        torch.tensor(np.array(all_c),     dtype=torch.float32),  # (N, T, 2)
    )


class EpisodeDataset(Dataset):
    def __init__(self, transitions, states, actions, next_states):
        self.transitions = transitions
        self.states      = states
        self.actions     = actions
        self.next_states = next_states

    def __len__(self): return len(self.transitions)

    def __getitem__(self, idx):
        return (self.transitions[idx], self.states[idx],
                self.actions[idx], self.next_states[idx])


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len          = 40
    context_duration = 20
    small_delta      = 3
    large_delta      = 21
    n_train          = 5000
    n_val            = 500
    batch_size       = 64
    epochs           = 30
    lr               = 1e-3
    lambda_nce       = 0.1    # weight on InfoNCE loss
    temperature      = 0.1     # InfoNCE temperature

    print(f"Device: {device}")
    print(f"Training sequences: {n_train} | Val sequences: {n_val}")
    print(f"Seq len: {seq_len} | Context duration: {context_duration} | Batch: {batch_size} | Epochs: {epochs}")
    print(f"λ_InfoNCE: {lambda_nce} | Temperature: {temperature}")
    print(f"InfoNCE small_delta: {small_delta} | large_delta: {large_delta}\n")

    # Generate long sequences — context changes every context_duration steps
    # EC never sees context, infers from temporal proximity of transitions
    print("Generating sequences...")
    tr = generate_sequences(n_train, seq_len=seq_len, context_duration=context_duration, seed=0)
    va = generate_sequences(n_val,   seq_len=seq_len, context_duration=context_duration, seed=99)

    train_ds = EpisodeDataset(tr[0], tr[1], tr[2], tr[3])
    val_ds   = EpisodeDataset(va[0], va[1], va[2], va[3])
    val_ctx  = va[4]   # true context — only for evaluation

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model     = EC_HC_CB().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_mse": [], "val_mse": [], "train_nce": [], "val_nce": []}

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_mse, total_nce = 0.0, 0.0

        for trans, s, a, ns in train_loader:
            trans, s, a, ns = trans.to(device), s.to(device), a.to(device), ns.to(device)

            s_pred, z = model(trans, s, a)

            loss_mse = F.mse_loss(s_pred, ns)
            loss_nce = infonce_loss(z, small_delta=small_delta, large_delta=large_delta, temperature=temperature)
            loss     = loss_mse + lambda_nce * loss_nce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mse += loss_mse.item() * len(trans)
            total_nce += loss_nce.item() * len(trans)

        total_mse /= len(train_ds)
        total_nce /= len(train_ds)
        history["train_mse"].append(total_mse)
        history["train_nce"].append(total_nce)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_mse, val_nce = 0.0, 0.0

        with torch.no_grad():
            for trans, s, a, ns in val_loader:
                trans, s, a, ns = trans.to(device), s.to(device), a.to(device), ns.to(device)
                s_pred, z = model(trans, s, a)
                val_mse += F.mse_loss(s_pred, ns).item() * len(trans)
                val_nce += infonce_loss(z, small_delta=small_delta, large_delta=large_delta, temperature=temperature).item() * len(trans)

        val_mse /= len(val_ds)
        val_nce /= len(val_ds)
        history["val_mse"].append(val_mse)
        history["val_nce"].append(val_nce)

        print(f"Epoch {epoch:02d} | "
              f"Train MSE: {total_mse:.5f}  NCE: {total_nce:.4f} | "
              f"Val MSE: {val_mse:.5f}  NCE: {val_nce:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "seq_len": seq_len,
        "lambda_nce": lambda_nce,
        "temperature": temperature,
    }, "checkpoints/ec_hc_cb.pt")
    print("\nSaved to checkpoints/ec_hc_cb.pt")

    # ── Evaluate: does z_t encode true context? ───────────────────────────────
    print("\nEvaluating context encoding in z_t...")

    model.eval()
    all_z_final = []

    with torch.no_grad():
        for i in range(0, len(val_ds), batch_size):
            trans = va[0][i:i+batch_size].to(device)
            s     = va[1][i:i+batch_size].to(device)
            a     = va[2][i:i+batch_size].to(device)
            _, z  = model(trans, s, a)
            all_z_final.append(z[:, -1, :].cpu())   # z_t at final timestep

    all_z_final = torch.cat(all_z_final).numpy()      # (n_val, 16)
    # val_ctx is (n_val, T, 2) — use last timestep's context for correlation
    true_ctx    = val_ctx[:, -1, :].numpy()            # (n_val, 2)

    import scipy.stats as stats
    r_beta  = max(abs(stats.pearsonr(all_z_final[:, d], true_ctx[:, 0])[0]) for d in range(16))
    r_gamma = max(abs(stats.pearsonr(all_z_final[:, d], true_ctx[:, 1])[0]) for d in range(16))
    print(f"Max |r| z_t vs true beta  : {r_beta:.4f}  (should be high if EC learned context)")
    print(f"Max |r| z_t vs true gamma : {r_gamma:.4f}")

    # ── Plot training curves ──────────────────────────────────────────────────
    os.makedirs("figs", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EC + HC + CB training (2 context blocks, context inferred via temporal contrastive loss)")

    axes[0].plot(history["train_mse"], label="train")
    axes[0].plot(history["val_mse"],   label="val")
    axes[0].set_title("MSE loss (prediction)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].plot(history["train_nce"], label="train")
    axes[1].plot(history["val_nce"],   label="val")
    axes[1].set_title("InfoNCE loss (context consistency)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("InfoNCE")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figs/ec_hc_cb_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figs/ec_hc_cb_training.png")


if __name__ == "__main__":
    t0 = time.time()
    train()
    print(f"\nTotal training time: {time.time() - t0:.1f}s")

"""
analyze_ec_similarity.py

Tests the temporal proximity hypothesis for EC's z_t vectors:

  <z_t, z_{t+Δt}>   is HIGH when Δt is small  (close in time → same context)
  <z_t, z_{t+Δt'}>  is LOW  when Δt' is large (far in time  → different context)

Loads trained checkpoint from checkpoints/ec_hc_cb.pt.
Falls back to random weights if checkpoint not found (gives a flat/noisy baseline).

Outputs:
  figs/ec_similarity_vs_delta_t.png  — mean sim vs Δt (with ±1 std band)
  figs/ec_similarity_matrix.png      — mean T×T heatmap (should be block-diagonal)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from train_ec_hc_cb import EC_HC_CB, generate_sequences

CHECKPOINT   = "checkpoints/ec_hc_cb.pt"
SEQ_LEN      = 60
CTX_DUR      = 20     # context changes every 20 steps → 3 blocks per sequence
N_SEQS       = 500    # number of sequences to average over
SEED         = 42

os.makedirs("figs", exist_ok=True)


# ── Load EC from checkpoint ───────────────────────────────────────────────────

def load_ec(checkpoint_path):
    model = EC_HC_CB()
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")
        trained = True
    else:
        print(f"WARNING: {checkpoint_path} not found — using random weights.")
        trained = False
    model.eval()
    return model.ec, trained


# ── Cosine similarity matrix for one sequence ─────────────────────────────────

def cosine_sim_matrix(z_seq):
    """
    z_seq : (T, D) tensor — z_t vectors for one sequence
    Returns (T, T) numpy array of pairwise cosine similarities
    """
    z_norm = F.normalize(z_seq, dim=-1)        # (T, D)
    return (z_norm @ z_norm.T).cpu().numpy()   # (T, T)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ec, trained = load_ec(CHECKPOINT)

    print(f"Generating {N_SEQS} sequences  (seq_len={SEQ_LEN}, ctx_dur={CTX_DUR})...")
    data = generate_sequences(N_SEQS, seq_len=SEQ_LEN,
                               context_duration=CTX_DUR, seed=SEED)
    transitions = data[0]   # (N, T, 6)

    print("Running EC forward pass...")
    with torch.no_grad():
        z, _ = ec(transitions)          # (N, T, latent_dim)

    N, T, D = z.shape
    print(f"  z shape: {z.shape}")

    # ── Compute per-sequence similarity matrices then average ─────────────────
    print("Computing similarity matrices...")
    mean_sim = np.zeros((T, T))

    # Also collect per-sequence mean-sim-by-Δt for error bars
    sim_by_delta = np.zeros((N, T))   # sim_by_delta[i, dt] = mean sim at lag dt in seq i

    for i in range(N):
        sm = cosine_sim_matrix(z[i])            # (T, T)
        mean_sim += sm
        for dt in range(T):
            # average over all anchors t such that t + dt < T
            sim_by_delta[i, dt] = sm.diagonal(offset=dt).mean()

    mean_sim /= N                               # (T, T) — average similarity matrix

    mean_by_delta = sim_by_delta.mean(axis=0)   # (T,)
    std_by_delta  = sim_by_delta.std(axis=0)    # (T,)
    delta_ts      = np.arange(T)

    # ── Plot 1: mean similarity vs Δt ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(delta_ts, mean_by_delta, color="steelblue", linewidth=2,
            label="mean cosine sim")
    ax.fill_between(delta_ts,
                    mean_by_delta - std_by_delta,
                    mean_by_delta + std_by_delta,
                    alpha=0.2, color="steelblue", label="±1 std (across seqs)")

    # Mark context block boundaries
    first = True
    for boundary in range(CTX_DUR, T, CTX_DUR):
        lbl = f"context boundary (Δt = {CTX_DUR})" if first else ""
        ax.axvline(boundary, color="red", linestyle="--", linewidth=1.2, label=lbl)
        first = False

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Temporal lag  Δt", fontsize=12)
    ax.set_ylabel("Mean cosine similarity\n⟨z_t , z_{t+Δt}⟩", fontsize=12)
    title = "EC temporal similarity" + (" (trained)" if trained else " (random weights)")
    ax.set_title(title + "\nClose in time → similar z_t", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, T - 1)

    plt.tight_layout()
    plt.savefig("figs/ec_similarity_vs_delta_t.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figs/ec_similarity_vs_delta_t.png")

    # ── Plot 2: mean T×T similarity heatmap ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean_sim, vmin=-1, vmax=1, cmap="RdBu_r",
                   aspect="auto", origin="upper")
    plt.colorbar(im, ax=ax, label="cosine similarity")

    # Draw context block boundaries on the heatmap
    for boundary in range(CTX_DUR, T, CTX_DUR):
        ax.axhline(boundary - 0.5, color="white", linewidth=1.5, linestyle="--")
        ax.axvline(boundary - 0.5, color="white", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Timestep  t'", fontsize=11)
    ax.set_ylabel("Timestep  t",  fontsize=11)
    ax.set_title(
        f"Mean ⟨z_t , z_t'⟩  (N={N} seqs,  {3} context blocks)\n"
        "Should be block-diagonal: high within block, low across",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("figs/ec_similarity_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figs/ec_similarity_matrix.png")

    # ── Summary stats ─────────────────────────────────────────────────────────
    within_block_deltas = delta_ts[delta_ts <= CTX_DUR // 2]       # Δt ≤ 10 (within block)
    across_block_deltas = delta_ts[delta_ts >= CTX_DUR + 5]        # Δt ≥ 25 (across boundary)

    sim_within = mean_by_delta[within_block_deltas].mean()
    sim_across = mean_by_delta[across_block_deltas].mean()

    print(f"\nSummary ({N} sequences, seq_len={T}, ctx_dur={CTX_DUR}):")
    print(f"  sim at Δt=1   (very close):    {mean_by_delta[1]:.4f}  ← should be HIGH")
    print(f"  sim at Δt=10  (within block):  {mean_by_delta[10]:.4f}")
    print(f"  sim at Δt=20  (at boundary):   {mean_by_delta[20]:.4f}")
    print(f"  sim at Δt=30  (across blocks): {mean_by_delta[30]:.4f}")
    print(f"  sim at Δt=50  (very far):      {mean_by_delta[50]:.4f}  ← should be LOW")
    print(f"\n  Mean within-block sim  (Δt ≤ {CTX_DUR//2}): {sim_within:.4f}")
    print(f"  Mean across-block sim  (Δt ≥ {CTX_DUR+5}):  {sim_across:.4f}")
    print(f"  Gap: {sim_within - sim_across:.4f}  "
          f"({'positive → temporal structure present' if sim_within > sim_across else 'NEGATIVE → no temporal structure'})")


if __name__ == "__main__":
    main()

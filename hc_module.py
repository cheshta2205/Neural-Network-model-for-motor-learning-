# hc_module.py
# Hippocampus (HC) as a slow consolidator of EC's latent context signal.
#
# Role: HC receives the fast-updating z_t from EC and integrates it slowly
# over time, producing a stable context signal c_t to send to CB.
# This implements the idea that HC consolidates episodic signals from EC
# into a slower and a more reliable representation.
#
# Mechanism: leaky integrator with a learnable smoothing factor alpha
#   c_t = (1 - alpha) * c_{t-1}  +  alpha * z_t
#
#   Small alpha → slow updates → HC changes gradually (more stable)
#   Large alpha → fast updates → HC tracks EC closely (less consolidation)
#
# Input  : z_t from EC     →  (batch, seq_len, latent_dim)
# Output : c_t sent to CB  →  (batch, seq_len, hc_dim)

import torch
import torch.nn as nn


class Hippocampus(nn.Module):
    def __init__(self, latent_dim=16, hc_dim=16):
        """
        Args:
            latent_dim : size of z_t coming in from EC
            hc_dim     : size of c_t going out to CB
                         kept same as latent_dim by default
        """
        super().__init__()

        self.hc_dim = hc_dim

        #Learnable smoothing factor 
        # alpha controls how fast HC consolidates EC's signal.
        # Stored as raw parameter, passed through sigmoid to keep in (0, 1).
        # Initialised at -2.2 so sigmoid(-2.2) ≈ 0.1 → starts slow.
        self.alpha_raw = nn.Parameter(torch.tensor(-2.2))

        # ── Input projection 
        # Projects z_t (from EC) into HC's own representational space
        # before integrating. Allows HC to learn what to consolidate.
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hc_dim),
            nn.Tanh(),
        )

        # ── Output projection
        # Projects HC's internal state c_t into the signal sent to CB.
        self.output_proj = nn.Linear(hc_dim, hc_dim)

    def forward(self, z_seq, c0=None):
        """
        Args:
            z_seq : (batch, seq_len, latent_dim) — z_t sequence from EC
            c0    : (batch, hc_dim) — optional initial HC state
                    if None, HC starts with no prior context (zeros)

        Returns:
            c_out  : (batch, seq_len, hc_dim) — consolidated context at each timestep
            c_last : (batch, hc_dim)          — final HC state (carry over to next episode)
        """
        batch, seq_len, _ = z_seq.shape

        # Initialise HC state
        c = torch.zeros(batch, self.hc_dim, device=z_seq.device) if c0 is None else c0

        # Smoothing factor: sigmoid keeps alpha in (0, 1)
        alpha = torch.sigmoid(self.alpha_raw)

        # Step through the sequence one timestep at a time
        # (leaky integrator cannot be parallelised like GRU — it's inherently sequential)
        c_seq = []
        for t in range(seq_len):
            # Project EC's z_t into HC space
            z_proj = self.input_proj(z_seq[:, t, :])   # (batch, hc_dim)

            # Leaky integration: blend old HC state with new EC signal
            # Small alpha → old state dominates → slow, stable consolidation
            # Large alpha → new signal dominates → fast updates like EC
            c = (1 - alpha) * c + alpha * z_proj        # (batch, hc_dim)

            c_seq.append(c)

        # Stack into sequence
        c_seq = torch.stack(c_seq, dim=1)               # (batch, seq_len, hc_dim)

        # Final output projection before sending to CB
        c_out = self.output_proj(c_seq)                 # (batch, seq_len, hc_dim)

        return c_out, c   # return sequence and final state


# ── Quick sanity check
if __name__ == "__main__":
    batch_size = 32
    seq_len    = 20
    latent_dim = 16    # same as EC's latent_dim
    hc_dim     = 16

    hc = Hippocampus(latent_dim=latent_dim, hc_dim=hc_dim)

    # Simulate z_t coming in from EC
    z_seq = torch.randn(batch_size, seq_len, latent_dim)

    c_out, c_last = hc(z_seq)

    print(f"EC input  (z_seq) shape : {z_seq.shape}")    # (32, 20, 16)
    print(f"HC output (c_out) shape : {c_out.shape}")    # (32, 20, 16)
    print(f"HC final  (c_last) shape: {c_last.shape}")   # (32, 16)

    # Print learned alpha to confirm it starts slow
    alpha = torch.sigmoid(hc.alpha_raw)
    print(f"\nLearned alpha (smoothing): {alpha.item():.4f}  (starts ~0.1 = slow)")
    print("\nHC module looks good.")

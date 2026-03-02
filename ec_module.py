# ec_module.py
# Entorhinal Cortex (EC) as a GRU-based RNN -

# Role: EC has episodic memory - observes the sequence of transitions (s_t, a_t, s_{t+1}) over time
# and integrates them to produce - latent context estimate z_t at each timestep. z_t 
# summarises "what kind of environment am I in/ what's my context?" from episodic experience.
#
# Input  per timestep : [s_t, a_t, s_{t+1}]  →  6 dims (x and y)
# Hidden state        : h_t                   →  hidden_dim (accumulates evidence/history)
# Output per timestep : z_t                   →  latent_dim (context, latent)

import torch
import torch.nn as nn


class EntorhinalCortex(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, latent_dim=16):
        """
        Args:
            input_dim  : size of input at each timestep
                         default 6 = [s_t(2), a_t(2), s_{t+1}(2)]
            hidden_dim : size of GRU hidden state
                         controls how much history EC can remember/ memory capacity of EC 
            latent_dim : size of output context vector z_t
                         this is what goes to HC as the input  (context as c)
        """
        super().__init__()

        # GRU
        # GRU takes the sequence of transitions and updates its hidden state h_t
        # at every timestep. h_t is EC's running belief about the context.
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,       # input shape: (batch, seq_len, input_dim)
        )

        # Readout layer 
        # Projects hidden state h_t down to the latent context z_t.
        # z_t is a lower-dimensional summary — what EC infers about context based on the history.
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),              # bounds z_t to [-1, 1]
        )

    def forward(self, transitions, h0=None):
        """
        Args:
            transitions : (batch, seq_len, 6)  — sequence of [s_t, a_t, s_{t+1}]
            h0          : (1, batch, hidden_dim) — optional initial hidden state
                          if None, starts from zeros (no prior context)

        Returns:
            z    : (batch, seq_len, latent_dim) — context estimate at each timestep
            h_last: (1, batch, hidden_dim)      — final hidden state
                    (can be passed as h0 for the next sequence chunk)
        """
        # Run GRU over the full sequence
        # h_seq: (batch, seq_len, hidden_dim) — hidden state at every timestep
        h_seq, h_last = self.gru(transitions, h0)

        # Project each hidden state to latent context z_t
        z = self.readout(h_seq)   # (batch, seq_len, latent_dim)

        return z, h_last


# sanity check
if __name__ == "__main__":
    batch_size = 32
    seq_len    = 20      # 20 timesteps of experience before making a prediction
    input_dim  = 6       # [s_t(2), a_t(2), s_{t+1}(2)] input state actions and next state 
    hidden_dim = 256
    latent_dim = 16

    ec = EntorhinalCortex(input_dim=input_dim,
                          hidden_dim=hidden_dim,
                          latent_dim=latent_dim)

    # Dummy sequence of transitions
    transitions = torch.randn(batch_size, seq_len, input_dim)

    z, h_last = ec(transitions)

    print(f"Input  shape : {transitions.shape}")   # (32, 20, 6)
    print(f"z_t    shape : {z.shape}")             # (32, 20, 16)
    print(f"h_last shape : {h_last.shape}")        # (1,  32, 256)
    print(f"z_t range    : [{z.min():.3f}, {z.max():.3f}]")  # should be in [-1,1]
    print("\nEC module looks good.")

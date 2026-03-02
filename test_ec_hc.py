import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.stats as stats
import os

from ec_module import EntorhinalCortex
from hc_module import Hippocampus

os.makedirs("figs/ec_hc_tests", exist_ok=True)

# ── Instantiate modules ──────

LATENT_DIM = 16
HIDDEN_DIM = 256
HC_DIM     = 16
SEQ_LEN    = 20

ec = EntorhinalCortex(input_dim=6, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
hc = Hippocampus(latent_dim=LATENT_DIM, hc_dim=HC_DIM)

ec.eval()
hc.eval()

print("EC and HC instantiated.")
print(f"  EC alpha (HC smoothing): {torch.sigmoid(hc.alpha_raw).item():.4f}  (should be ~0.1)")


# ── Helper: generate fixed-context episodes ───────────────────────────────────
# Unlike data.py which samples a new context every timestep,
# here each episode has one fixed beta and gamma throughout.
# This lets EC accumulate evidence about a stable context.

def generate_fixed_context_episodes(n_episodes, seq_len, beta, gamma, seed=0):
    """
    Returns transitions: (n_episodes, seq_len, 6)
    Each episode has fixed beta, gamma throughout.
    """
    rng = np.random.default_rng(seed)
    actions = [(-1., 0.), (1., 0.), (0., -1.), (0., 1.)]

    episodes = []
    for _ in range(n_episodes):
        x, y = float(rng.integers(0, 5)), float(rng.integers(0, 5))
        seq = []
        for _ in range(seq_len):
            dx, dy = actions[rng.integers(0, 4)]
            a = np.array([dx, dy])
            a_norm = np.linalg.norm(a)
            a_hat = a / a_norm if a_norm > 0 else np.zeros(2)
            a_hat_perp = np.array([-a_hat[1], a_hat[0]])

            f_env = beta * a_hat + gamma * a_hat_perp
            x_next = float(np.clip(x + dx + f_env[0], 0, 4))
            y_next = float(np.clip(y + dy + f_env[1], 0, 4))

            seq.append([x, y, dx, dy, x_next, y_next])
            x, y = x_next, y_next

        episodes.append(seq)

    return torch.tensor(np.array(episodes, dtype=np.float32))  # (n_episodes, seq_len, 6)


# ══════════
# LEVEL 1: Sanity checks — shapes and ranges
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LEVEL 1: Sanity checks")
print("="*60)

batch_size = 32
dummy_seq  = torch.randn(batch_size, SEQ_LEN, 6)

with torch.no_grad():
    z, h_last = ec(dummy_seq)
    c_out, c_last = hc(z)

print(f"EC input  shape : {dummy_seq.shape}")      # (32, 20, 6)
print(f"EC z      shape : {z.shape}")              # (32, 20, 16)
print(f"EC h_last shape : {h_last.shape}")         # (1,  32, 128)
print(f"HC c_out  shape : {c_out.shape}")          # (32, 20, 16)
print(f"HC c_last shape : {c_last.shape}")         # (32, 16)
print(f"z_t range       : [{z.min():.3f}, {z.max():.3f}]  (should be in [-1, 1])")

assert z.shape     == (batch_size, SEQ_LEN, LATENT_DIM), "EC output shape wrong"
assert h_last.shape == (1, batch_size, HIDDEN_DIM),       "EC h_last shape wrong"
assert c_out.shape  == (batch_size, SEQ_LEN, HC_DIM),     "HC output shape wrong"
assert z.min() >= -1.0 and z.max() <= 1.0,                "z_t out of [-1, 1]"

print("All Level 1 checks passed.")


# ════════════════════════════════════════════════════════════════════════════════
# LEVEL 2: HC smoothing — c_t should be smoother (lower variance) than z_t
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LEVEL 2: HC smoothing test")
print("="*60)

# Use a noisy z_seq and check that HC output has lower temporal variance
noisy_z = torch.randn(64, SEQ_LEN, LATENT_DIM)  # (64, 20, 16)

with torch.no_grad():
    c_out_smooth, _ = hc(noisy_z)

# Temporal variance: how much does each signal change step to step?
z_diff = noisy_z[:, 1:, :] - noisy_z[:, :-1, :]      # (64, 19, 16)
c_diff = c_out_smooth[:, 1:, :] - c_out_smooth[:, 1:, :]  # (64, 19, 16)

z_temporal_var = z_diff.var().item()
c_temporal_var = c_diff.var().item()

# Overall variance
z_var = noisy_z.var().item()
c_var = c_out_smooth.var().item()

print(f"z_t  temporal variance (step-to-step change): {z_temporal_var:.4f}")
print(f"c_t  temporal variance (step-to-step change): {c_temporal_var:.4f}")
print(f"c_t is {'smoother' if c_temporal_var < z_temporal_var else 'NOT smoother'} than z_t")
print(f"\nHC alpha (smoothing factor): {torch.sigmoid(hc.alpha_raw).item():.4f}")
print(f"  → c_t = {1-torch.sigmoid(hc.alpha_raw).item():.2f} * c_{{t-1}} + {torch.sigmoid(hc.alpha_raw).item():.2f} * z_t")

# Plot: z_t vs c_t over time for one batch, one latent dim
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Level 2: HC smoothing — z_t vs c_t over time", fontsize=12)

t = np.arange(SEQ_LEN)
for dim in range(4):  # plot first 4 latent dims
    axes[0].plot(t, noisy_z[0, :, dim].numpy(), alpha=0.7)
    axes[1].plot(t, c_out_smooth[0, :, dim].numpy(), alpha=0.7)

axes[0].set_title(f"z_t from EC (var={z_var:.3f})")
axes[0].set_xlabel("Timestep")
axes[0].set_ylabel("Value")
axes[1].set_title(f"c_t from HC (should be smoother)")
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Value")

plt.tight_layout()
plt.savefig("figs/ec_hc_tests/level2_hc_smoothing.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figs/ec_hc_tests/level2_hc_smoothing.png")


# ════════════════════════════════════════════════════════════════════════════════
# LEVEL 3: Context discrimination
# Can z_t / c_t separate episodes from different contexts?
# NOTE: EC is untrained here — with trained weights this would be stronger.
#       This test shows the architecture's capacity to discriminate.
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LEVEL 3: Context discrimination")
print("="*60)

N_EPISODES = 100

# Generate episodes from 5 different beta values, gamma=0
beta_values  = [-0.5, -0.25, 0.0, 0.25, 0.5]
all_z_last   = []  # z_t at final timestep (t=20) per episode
all_c_last   = []  # c_t at final timestep
all_betas    = []

for beta in beta_values:
    episodes = generate_fixed_context_episodes(N_EPISODES, SEQ_LEN, beta=beta, gamma=0.0)

    with torch.no_grad():
        z, _ = ec(episodes)          # (100, 20, 16)
        c, _ = hc(z)                 # (100, 20, 16)

    all_z_last.append(z[:, -1, :].numpy())   # final timestep
    all_c_last.append(c[:, -1, :].numpy())
    all_betas.extend([beta] * N_EPISODES)

all_z_last = np.vstack(all_z_last)   # (500, 16)
all_c_last = np.vstack(all_c_last)   # (500, 16)
all_betas  = np.array(all_betas)

# PCA on z_t at final timestep, coloured by beta
pca_z = PCA(n_components=2).fit_transform(all_z_last)
pca_c = PCA(n_components=2).fit_transform(all_c_last)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Level 3: Context discrimination — PCA of z_t and c_t at t=20\n"
             "(EC untrained — trains weights would give stronger separation)", fontsize=11)

for ax, pca_data, label in zip(axes, [pca_z, pca_c], ["z_t (EC output)", "c_t (HC output)"]):
    sc = ax.scatter(pca_data[:, 0], pca_data[:, 1],
                    c=all_betas, cmap="coolwarm", s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="beta")
    ax.set_title(label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

plt.tight_layout()
plt.savefig("figs/ec_hc_tests/level3_context_discrimination.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figs/ec_hc_tests/level3_context_discrimination.png")

# Quantify: correlation of PC1 with beta
r_z, _ = stats.pearsonr(pca_z[:, 0], all_betas)
r_c, _ = stats.pearsonr(pca_c[:, 0], all_betas)
print(f"Correlation of z_t PC1 with beta: {r_z:.4f}")
print(f"Correlation of c_t PC1 with beta: {r_c:.4f}")
print("(With trained EC, these should be high)")


# ════════════════════════════════════════════════════════════════════════════════
# LEVEL 4: Does z_t improve over timesteps?
# Correlation of z_t with true beta should increase as EC sees more transitions
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("LEVEL 4: z_t improvement over timesteps")
print("="*60)

# Use all episodes from Level 3 but keep all timesteps
all_z_seq = []   # (n_contexts * N_EPISODES, seq_len, 16)
all_betas2 = []

for beta in beta_values:
    episodes = generate_fixed_context_episodes(N_EPISODES, SEQ_LEN, beta=beta, gamma=0.0, seed=42)
    with torch.no_grad():
        z, _ = ec(episodes)   # (100, 20, 16)
    all_z_seq.append(z.numpy())
    all_betas2.extend([beta] * N_EPISODES)

all_z_seq  = np.vstack(all_z_seq)   # (500, 20, 16)
all_betas2 = np.array(all_betas2)

# At each timestep t: compute max |correlation| between any z dim and beta
corr_over_time = []
for t in range(SEQ_LEN):
    z_t = all_z_seq[:, t, :]   # (500, 16)
    max_corr = max(abs(stats.pearsonr(z_t[:, dim], all_betas2)[0])
                   for dim in range(LATENT_DIM))
    corr_over_time.append(max_corr)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, SEQ_LEN + 1), corr_over_time, "o-", color="steelblue")
ax.set_xlabel("Timestep (number of transitions seen by EC)")
ax.set_ylabel("Max |correlation| of z_t dim with beta")
ax.set_title("Level 4: Does z_t get more informative about context over time?\n"
             "(EC untrained — with trained weights correlation should rise over timesteps)")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="r=0.5 reference")
ax.legend()
plt.tight_layout()
plt.savefig("figs/ec_hc_tests/level4_zt_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figs/ec_hc_tests/level4_zt_over_time.png")
print(f"Max correlation at t=1  : {corr_over_time[0]:.4f}")
print(f"Max correlation at t=10 : {corr_over_time[9]:.4f}")
print(f"Max correlation at t=20 : {corr_over_time[-1]:.4f}")
print("(With trained EC these should increase monotonically)")

print("\nAll tests complete.")

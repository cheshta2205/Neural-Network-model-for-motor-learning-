import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import scipy.stats as stats

from model import CerebellumForwardModelContinuous
from data import simulate_state_action_and_next_state
from torch.utils.data import Dataset, DataLoader


# Dataset (same as train.py)

class ForwardDatasetContinuous(Dataset):
    def __init__(self, states_xy, actions_vec, contexts, next_states_xy):
        self.states = torch.tensor(states_xy, dtype=torch.float32)
        self.actions = torch.tensor(actions_vec, dtype=torch.float32)
        self.contexts = torch.tensor(contexts, dtype=torch.float32)
        self.next_states = torch.tensor(next_states_xy, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.contexts[idx], self.next_states[idx]


# Create folder to save plots
save_dir = "representations"
os.makedirs(save_dir, exist_ok=True)  # creates folder if it doesn't exist


# Load model weights 

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(
    "checkpoints/cerebellum_forward_continuous_context_beta_gamma.pt",
    map_location=device
)

model = CerebellumForwardModelContinuous(input_dim=6, hidden_dim=256).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Model loaded.")

# Generate new data, on randomseed 123 

states_xy, actions_vec, next_states_xy, contexts = simulate_state_action_and_next_state(
    n_runs=100,
    T=200,
    H=ckpt["H"],
    W=ckpt["W"],
    action_set=ckpt["action_set"],
    slip_prob=ckpt["slip_prob"],
    seed=123,               # new seed
    context_mask=ckpt["context_mask"],
)

dataset = ForwardDatasetContinuous(states_xy, actions_vec, contexts, next_states_xy)
loader = DataLoader(dataset, batch_size=256)


# Capture hidden layers

hidden_1 = []
hidden_2 = []

def hook_h1(module, input, output):
    hidden_1.append(output.detach().cpu())

def hook_h2(module, input, output):
    hidden_2.append(output.detach().cpu())

handle1 = model.net[1].register_forward_hook(hook_h1)
handle2 = model.net[3].register_forward_hook(hook_h2)


# Run model to collect hidden activations

all_contexts = []
all_actions = []

with torch.no_grad():
    for s, a, c, y in loader:
        s, a, c = s.to(device), a.to(device), c.to(device)
        _ = model(s, a, c)

        all_contexts.append(c.cpu())
        all_actions.append(a.cpu())

handle1.remove()
handle2.remove()

H1 = torch.cat(hidden_1, dim=0).numpy()
H2 = torch.cat(hidden_2, dim=0).numpy()
C = torch.cat(all_contexts, dim=0).numpy()
A = torch.cat(all_actions, dim=0).numpy()

print("Hidden representations collected.")
print("H2 shape:", H2.shape)


# PCA Visualization- to understand variance distribution - only plotting 5000 points to speed figs up
# Subsample for PCA fitting (for speed purposes)
idx = np.random.choice(len(H2), size=5000, replace=False)

pca = PCA(n_components=2)
pca.fit(H2[idx])          # fit on 5k points only

H2_2d = pca.transform(H2) # transform full dataset

plot_idx = np.random.choice(len(H2_2d), size=3000, replace=False)

print("Plotting PCA colored by beta")
plt.figure()
plt.scatter(
    H2_2d[plot_idx, 0],
    H2_2d[plot_idx, 1],
    c=C[plot_idx, 0],
    cmap="coolwarm",
    s=4,
    alpha=0.6,
    edgecolors="none",
)
plt.colorbar(label="beta")
plt.title("PCA - colored by beta representations")
plt.savefig(os.path.join(save_dir, "pca_beta.png"), bbox_inches="tight")
plt.close()

print("Plotting PCA colored by gamma")
plt.figure()
plt.scatter(
    H2_2d[plot_idx, 0],
    H2_2d[plot_idx, 1],
    c=C[plot_idx, 1],
    cmap="viridis",
    s=4,
    alpha=0.6,
    edgecolors="none",
)
plt.colorbar(label="gamma")
plt.title("PCA - colored by gamma representations")
plt.savefig(os.path.join(save_dir, "pca_gamma.png"), bbox_inches="tight")
plt.close()


# Neuron Selectivity (Correlation)

beta_corr = []
gamma_corr = []

for i in range(H2.shape[1]):
    r_beta, _ = stats.pearsonr(H2[:, i], C[:, 0])
    r_gamma, _ = stats.pearsonr(H2[:, i], C[:, 1])
    beta_corr.append(r_beta)
    gamma_corr.append(r_gamma)

beta_corr = np.array(beta_corr)
gamma_corr = np.array(gamma_corr)

print("Plotting correlations of neurons with beta")
plt.figure()
plt.hist(beta_corr, bins=40)
plt.title("Neuron correlation with beta")
plt.savefig(os.path.join(save_dir, "hist_beta_corr.png"), bbox_inches="tight")
plt.close()

print("Plotting correlations of neurons with gamma")
plt.figure()
plt.hist(gamma_corr, bins=40)
plt.title("Neuron correlation with gamma")
plt.savefig(os.path.join(save_dir, "hist_gamma_corr.png"), bbox_inches="tight")
plt.close()

print("Max |beta correlation|:", np.max(np.abs(beta_corr)))
print("Max |gamma correlation|:", np.max(np.abs(gamma_corr)))


# State Selectivity (Correlation with x and y)

print("Computing neuron correlations with state (x, y)")

# We need states again (flattened like H2)
all_states = []

with torch.no_grad():
    for s, a, c, y in loader:
        all_states.append(s.cpu())

S = torch.cat(all_states, dim=0).numpy()  # shape: (N_samples, 2)


# PCA Visualization - colored by state and action
print("Plotting PCA colored by state_x")
plt.figure()
plt.scatter(
    H2_2d[plot_idx, 0],
    H2_2d[plot_idx, 1],
    c=S[plot_idx, 0],
    cmap="plasma",
    s=4,
    alpha=0.6,
    edgecolors="none",
)
plt.colorbar(label="state_x")
plt.title("PCA - colored by state_x")
plt.savefig(os.path.join(save_dir, "pca_state_x.png"), bbox_inches="tight")
plt.close()

print("Plotting PCA colored by state_y")
plt.figure()
plt.scatter(
    H2_2d[plot_idx, 0],
    H2_2d[plot_idx, 1],
    c=S[plot_idx, 1],
    cmap="plasma",
    s=4,
    alpha=0.6,
    edgecolors="none",
)
plt.colorbar(label="state_y")
plt.title("PCA - colored by state_y")
plt.savefig(os.path.join(save_dir, "pca_state_y.png"), bbox_inches="tight")
plt.close()

print("Plotting PCA colored by action dx")
plt.figure()
plt.scatter(
    H2_2d[plot_idx, 0],
    H2_2d[plot_idx, 1],
    c=A[plot_idx, 0],
    cmap="coolwarm",
    s=4,
    alpha=0.6,
    edgecolors="none",
)
plt.colorbar(label="action dx")
plt.title("PCA - colored by action dx")
plt.savefig(os.path.join(save_dir, "pca_action_dx.png"), bbox_inches="tight")
plt.close()

print("Plotting PCA colored by action dy")
plt.figure()
plt.scatter(
    H2_2d[plot_idx, 0],
    H2_2d[plot_idx, 1],
    c=A[plot_idx, 1],
    cmap="coolwarm",
    s=4,
    alpha=0.6,
    edgecolors="none",
)
plt.colorbar(label="action dy")
plt.title("PCA - colored by action dy")
plt.savefig(os.path.join(save_dir, "pca_action_dy.png"), bbox_inches="tight")
plt.close()


state_x_corr = []
state_y_corr = []

for i in range(H2.shape[1]):
    r_x, _ = stats.pearsonr(H2[:, i], S[:, 0])
    r_y, _ = stats.pearsonr(H2[:, i], S[:, 1])
    state_x_corr.append(r_x)
    state_y_corr.append(r_y)

state_x_corr = np.array(state_x_corr)
state_y_corr = np.array(state_y_corr)

plt.figure()
plt.hist(state_x_corr, bins=40)
plt.title("Neuron correlation with state_x")
plt.savefig(os.path.join(save_dir, "hist_state_x_corr.png"), bbox_inches="tight")
plt.close()

plt.figure()
plt.hist(state_y_corr, bins=40)
plt.title("Neuron correlation with state_y")
plt.savefig(os.path.join(save_dir, "hist_state_y_corr.png"), bbox_inches="tight")
plt.close()

print("Max |state_x correlation|:", np.max(np.abs(state_x_corr)))
print("Max |state_y correlation|:", np.max(np.abs(state_y_corr)))


# Action Selectivity


print("Computing neuron correlations with actions")

# If actions are one-hot encoded
n_actions = A.shape[1]

for action_idx in range(n_actions):
    action_corr = []
    
    for i in range(H2.shape[1]):
        r, _ = stats.pearsonr(H2[:, i], A[:, action_idx])
        action_corr.append(r)
    
    action_corr = np.array(action_corr)
    
    plt.figure()
    plt.hist(action_corr, bins=40)
    plt.title(f"Neuron correlation with action_{action_idx}")
    plt.savefig(os.path.join(save_dir, f"hist_action_{action_idx}_corr.png"),
                bbox_inches="tight")
    plt.close()
    
    print(f"Max |action_{action_idx} correlation|:",
          np.max(np.abs(action_corr)))


# ── Layer 1 Analysis ──────────────────────────────────────────────────────────

save_dir_l1 = os.path.join(save_dir, "representations_layer1")
os.makedirs(save_dir_l1, exist_ok=True)

print("\n--- Layer 1 Analysis ---")
print("H1 shape:", H1.shape)

# PCA on H1
idx_l1 = np.random.choice(len(H1), size=5000, replace=False)
pca_l1 = PCA(n_components=2)
pca_l1.fit(H1[idx_l1])
H1_2d = pca_l1.transform(H1)

plot_idx_l1 = np.random.choice(len(H1_2d), size=3000, replace=False)

# PCA colored by context
print("L1: Plotting PCA colored by beta")
plt.figure()
plt.scatter(H1_2d[plot_idx_l1, 0], H1_2d[plot_idx_l1, 1],
            c=C[plot_idx_l1, 0], cmap="coolwarm", s=4, alpha=0.6, edgecolors="none")
plt.colorbar(label="beta")
plt.title("L1 PCA - colored by beta")
plt.savefig(os.path.join(save_dir_l1, "pca_beta.png"), bbox_inches="tight")
plt.close()

print("L1: Plotting PCA colored by gamma")
plt.figure()
plt.scatter(H1_2d[plot_idx_l1, 0], H1_2d[plot_idx_l1, 1],
            c=C[plot_idx_l1, 1], cmap="viridis", s=4, alpha=0.6, edgecolors="none")
plt.colorbar(label="gamma")
plt.title("L1 PCA - colored by gamma")
plt.savefig(os.path.join(save_dir_l1, "pca_gamma.png"), bbox_inches="tight")
plt.close()

# PCA colored by state and action
print("L1: Plotting PCA colored by state_x")
plt.figure()
plt.scatter(H1_2d[plot_idx_l1, 0], H1_2d[plot_idx_l1, 1],
            c=S[plot_idx_l1, 0], cmap="plasma", s=4, alpha=0.6, edgecolors="none")
plt.colorbar(label="state_x")
plt.title("L1 PCA - colored by state_x")
plt.savefig(os.path.join(save_dir_l1, "pca_state_x.png"), bbox_inches="tight")
plt.close()

print("L1: Plotting PCA colored by state_y")
plt.figure()
plt.scatter(H1_2d[plot_idx_l1, 0], H1_2d[plot_idx_l1, 1],
            c=S[plot_idx_l1, 1], cmap="plasma", s=4, alpha=0.6, edgecolors="none")
plt.colorbar(label="state_y")
plt.title("L1 PCA - colored by state_y")
plt.savefig(os.path.join(save_dir_l1, "pca_state_y.png"), bbox_inches="tight")
plt.close()

print("L1: Plotting PCA colored by action dx")
plt.figure()
plt.scatter(H1_2d[plot_idx_l1, 0], H1_2d[plot_idx_l1, 1],
            c=A[plot_idx_l1, 0], cmap="coolwarm", s=4, alpha=0.6, edgecolors="none")
plt.colorbar(label="action dx")
plt.title("L1 PCA - colored by action dx")
plt.savefig(os.path.join(save_dir_l1, "pca_action_dx.png"), bbox_inches="tight")
plt.close()

print("L1: Plotting PCA colored by action dy")
plt.figure()
plt.scatter(H1_2d[plot_idx_l1, 0], H1_2d[plot_idx_l1, 1],
            c=A[plot_idx_l1, 1], cmap="coolwarm", s=4, alpha=0.6, edgecolors="none")
plt.colorbar(label="action dy")
plt.title("L1 PCA - colored by action dy")
plt.savefig(os.path.join(save_dir_l1, "pca_action_dy.png"), bbox_inches="tight")
plt.close()

# Correlation histograms — context
beta_corr_l1, gamma_corr_l1 = [], []
for i in range(H1.shape[1]):
    r_b, _ = stats.pearsonr(H1[:, i], C[:, 0])
    r_g, _ = stats.pearsonr(H1[:, i], C[:, 1])
    beta_corr_l1.append(r_b)
    gamma_corr_l1.append(r_g)
beta_corr_l1  = np.array(beta_corr_l1)
gamma_corr_l1 = np.array(gamma_corr_l1)

plt.figure()
plt.hist(beta_corr_l1, bins=40)
plt.title("L1 Neuron correlation with beta")
plt.savefig(os.path.join(save_dir_l1, "hist_beta_corr.png"), bbox_inches="tight")
plt.close()

plt.figure()
plt.hist(gamma_corr_l1, bins=40)
plt.title("L1 Neuron correlation with gamma")
plt.savefig(os.path.join(save_dir_l1, "hist_gamma_corr.png"), bbox_inches="tight")
plt.close()

print("L1 Max |beta correlation|:",  np.max(np.abs(beta_corr_l1)))
print("L1 Max |gamma correlation|:", np.max(np.abs(gamma_corr_l1)))

# Correlation histograms — state
state_x_corr_l1, state_y_corr_l1 = [], []
for i in range(H1.shape[1]):
    r_x, _ = stats.pearsonr(H1[:, i], S[:, 0])
    r_y, _ = stats.pearsonr(H1[:, i], S[:, 1])
    state_x_corr_l1.append(r_x)
    state_y_corr_l1.append(r_y)
state_x_corr_l1 = np.array(state_x_corr_l1)
state_y_corr_l1 = np.array(state_y_corr_l1)

plt.figure()
plt.hist(state_x_corr_l1, bins=40)
plt.title("L1 Neuron correlation with state_x")
plt.savefig(os.path.join(save_dir_l1, "hist_state_x_corr.png"), bbox_inches="tight")
plt.close()

plt.figure()
plt.hist(state_y_corr_l1, bins=40)
plt.title("L1 Neuron correlation with state_y")
plt.savefig(os.path.join(save_dir_l1, "hist_state_y_corr.png"), bbox_inches="tight")
plt.close()

print("L1 Max |state_x correlation|:", np.max(np.abs(state_x_corr_l1)))
print("L1 Max |state_y correlation|:", np.max(np.abs(state_y_corr_l1)))

# Correlation histograms — actions
for action_idx in range(A.shape[1]):
    action_corr_l1 = []
    for i in range(H1.shape[1]):
        r, _ = stats.pearsonr(H1[:, i], A[:, action_idx])
        action_corr_l1.append(r)
    action_corr_l1 = np.array(action_corr_l1)
    plt.figure()
    plt.hist(action_corr_l1, bins=40)
    plt.title(f"L1 Neuron correlation with action_{action_idx}")
    plt.savefig(os.path.join(save_dir_l1, f"hist_action_{action_idx}_corr.png"),
                bbox_inches="tight")
    plt.close()
    print(f"L1 Max |action_{action_idx} correlation|:", np.max(np.abs(action_corr_l1)))
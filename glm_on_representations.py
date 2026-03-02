import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict

from model import CerebellumForwardModelContinuous
from data import simulate_state_action_and_next_state
from torch.utils.data import Dataset, DataLoader


# ── Dataset 

class ForwardDatasetContinuous(Dataset):
    def __init__(self, states_xy, actions_vec, contexts, next_states_xy):
        self.states      = torch.tensor(states_xy,      dtype=torch.float32)
        self.actions     = torch.tensor(actions_vec,    dtype=torch.float32)
        self.contexts    = torch.tensor(contexts,       dtype=torch.float32)
        self.next_states = torch.tensor(next_states_xy, dtype=torch.float32)

    def __len__(self): return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.contexts[idx], self.next_states[idx]


# ── Load model 
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(
    "checkpoints/cerebellum_forward_continuous_context_beta_gamma.pt",
    map_location=device,
)

model = CerebellumForwardModelContinuous(input_dim=6, hidden_dim=256).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Model loaded.")


# ── Generate data , seed 123
states_xy, actions_vec, next_states_xy, contexts = simulate_state_action_and_next_state(
    n_runs=100,
    T=200,
    H=ckpt["H"],
    W=ckpt["W"],
    action_set=ckpt["action_set"],
    slip_prob=ckpt["slip_prob"],
    seed=123,
    context_mask=ckpt["context_mask"],
)

dataset = ForwardDatasetContinuous(states_xy, actions_vec, contexts, next_states_xy)
loader  = DataLoader(dataset, batch_size=256, shuffle=False)

print(f"Dataset size: {len(dataset)} samples")


# ── Extract hidden layer activations via hooks ──────────────────────────────

hidden_1, hidden_2 = [], []

def hook_h1(module, input, output): hidden_1.append(output.detach().cpu())
def hook_h2(module, input, output): hidden_2.append(output.detach().cpu())

handle1 = model.net[1].register_forward_hook(hook_h1)
handle2 = model.net[3].register_forward_hook(hook_h2)

all_s, all_a, all_c = [], [], []

with torch.no_grad():
    for s, a, c, y in loader:
        s, a, c = s.to(device), a.to(device), c.to(device)
        _ = model(s, a, c)
        all_s.append(s.cpu())
        all_a.append(a.cpu())
        all_c.append(c.cpu())

handle1.remove()
handle2.remove()

H1 = torch.cat(hidden_1, dim=0).numpy()   # (N, 256)
H2 = torch.cat(hidden_2, dim=0).numpy()   # (N, 256)
S  = torch.cat(all_s,    dim=0).numpy()   # (N, 2)
A  = torch.cat(all_a,    dim=0).numpy()   # (N, 2)
C  = torch.cat(all_c,    dim=0).numpy()   # (N, 2)

print(f"H1: {H1.shape}, H2: {H2.shape}")


# ── Build predictor sets ───────────────────────────────────────────────────

CA = np.hstack([
    C[:, 0:1] * A[:, 0:1],   # beta  * dx - context
    C[:, 0:1] * A[:, 1:2],   # beta  * dy - context
    C[:, 1:2] * A[:, 0:1],   # gamma * dx - context
    C[:, 1:2] * A[:, 1:2],   # gamma * dy - context
])

predictor_sets = {
    "Full (s+a+c+ca)":      np.hstack([S, A, C, CA]),
    "Drop state (a+c+ca)":  np.hstack([A, C, CA]),
    "Drop action (s+c)":    np.hstack([S, C]),
    "Drop context (s+a)":   np.hstack([S, A]),
    "Only state (s)":       S,
    "Only action (a)":      A,
    "Only context (c)":     C,
}


# ── Ridge pipeline  - regression to understand how well each of s or a or c can explain
# variance in neuronal firing (activations)
# Ridge natively handles multi-output Y (N, 256).
# cross_val_score with scoring="r2" returns mean R² across all 256 neurons.

def ridge_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])


# ── GLM variance partitioning on layer representations 
os.makedirs("figs", exist_ok=True)

layers = {"Layer 1 (H1)": H1, "Layer 2 (H2)": H2}
all_results = {}

for layer_name, H in layers.items():
    print(f"\n=== {layer_name} ===")
    results = {}

    for name, X in predictor_sets.items():
        r2 = cross_val_score(ridge_pipeline(), X, H, cv=5, scoring="r2").mean()
        results[name] = r2
        print(f"  {name}: mean R² = {r2:.4f}")

    all_results[layer_name] = results

    # Unique contributions
    full_r2   = results["Full (s+a+c+ca)"]
    drop_labels = ["state", "action", "context"]
    drop_keys   = ["Drop state (a+c+ca)", "Drop action (s+c)", "Drop context (s+a)"]
    drop_vals   = [full_r2 - results[k] for k in drop_keys]

    print(f"\n  Unique contributions (drop in R² from full):")
    for label, drop in zip(drop_labels, drop_vals):
        print(f"    {label}: {drop:.4f}")


# ── Plot ──────────────────────────────────────────────────────────────────────

names = list(predictor_sets.keys())

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Panel 1 & 2: Mean R² per predictor set, one panel per layer
for ax, (layer_name, results) in zip(axes[:2], all_results.items()):
    full_r2 = results["Full (s+a+c+ca)"]
    r2_vals = [results[n] for n in names]

    colors = (
        ["steelblue"] +      # Full
        ["grey"] * 3 +       # Drop variants
        ["darkorange"] * 3   # Only variants
    )

    bars = ax.bar(names, r2_vals, color=colors)
    ax.axhline(full_r2, color="red", linestyle="--", linewidth=1, label="Full model")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean R² across neurons")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Variance in {layer_name} explained by s / a / c")
    ax.legend()
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", fontsize=7)

# Panel 3: Unique contributions — both layers side by side
drop_labels = ["state", "action", "context"]
drop_keys   = ["Drop state (a+c+ca)", "Drop action (s+c)", "Drop context (s+a)"]
x     = np.arange(len(drop_labels))
width = 0.3

for i, (layer_name, results) in enumerate(all_results.items()):
    full_r2   = results["Full (s+a+c+ca)"]
    drop_vals = [full_r2 - results[k] for k in drop_keys]
    offset    = (i - 0.5) * width
    bars = axes[2].bar(x + offset, drop_vals, width, label=layer_name)
    for bar, val in zip(bars, drop_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                     f"{val:.3f}", ha="center", fontsize=7)

axes[2].set_xticks(x)
axes[2].set_xticklabels(drop_labels)
axes[2].set_ylabel("Unique contribution (drop in R² from full)")
axes[2].set_title("Unique contribution of each variable\n(H1 vs H2)")
axes[2].legend()
axes[2].set_ylim(0, axes[2].get_ylim()[1] * 1.2)

plt.tight_layout()
plt.savefig("figs/glm_on_representations.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nSaved to figs/glm_on_representations.png")


# ── Per-neuron heatmap ──────
# For each predictor set, use cross_val_predict to get held-out predictions
# for all 256 neurons in one pass (not 256 separate fits).
# Then compute R² per neuron from predictions vs actuals.

def r2_per_neuron(H, X):
    """5-fold CV predictions → R² for each of the 256 neurons."""
    H_pred = cross_val_predict(ridge_pipeline(), X, H, cv=5)
    ss_res  = ((H - H_pred) ** 2).sum(axis=0)          # (256,)
    ss_tot  = ((H - H.mean(axis=0)) ** 2).sum(axis=0)  # (256,)
    return 1 - ss_res / (ss_tot + 1e-10)               # (256,)


drop_keys_map = {
    "Full":    "Full (s+a+c+ca)",
    "NoState": "Drop state (a+c+ca)",
    "NoAct":   "Drop action (s+c)",
    "NoCtx":   "Drop context (s+a)",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 10))

for ax, (layer_name, H) in zip(axes, layers.items()):
    print(f"\nComputing per-neuron R² for {layer_name}...")

    r2_s = r2_per_neuron(H, predictor_sets["Only state (s)"])    # (256,)
    r2_a = r2_per_neuron(H, predictor_sets["Only action (a)"])   # (256,)
    r2_c = r2_per_neuron(H, predictor_sets["Only context (c)"])  # (256,)

    # Stack into matrix (256, 3): cols = [only_s, only_a, only_c]
    mat = np.stack([r2_s, r2_a, r2_c], axis=1)
    mat = np.clip(mat, 0, 1)

    # Sort neurons: group by dominant variable, then by magnitude within group
    dominant  = np.argmax(mat, axis=1)           # 0=state, 1=action, 2=context
    magnitude = mat[np.arange(256), dominant]
    sort_idx  = np.lexsort((magnitude, dominant))[::-1]

    mat_sorted = mat[sort_idx]

    im = ax.imshow(mat_sorted, aspect="auto", cmap="Reds", vmin=0, vmax=mat.max())
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["only state", "only action", "only context"], fontsize=10)
    ax.set_ylabel("Neuron (sorted by dominant variable)", fontsize=10)
    ax.set_title(f"{layer_name}\nR² per neuron (only s / only a / only c)", fontsize=11)
    plt.colorbar(im, ax=ax, label="R²")

    # Draw dividing lines between dominant-variable groups
    group_sizes = [(dominant == g).sum() for g in [0, 1, 2]]
    cum = 0
    for gs in group_sizes[:-1]:
        cum += gs
        ax.axhline(cum - 0.5, color="white", linewidth=1.5)

    # Label groups on y-axis
    cum = 0
    for gs, label in zip(group_sizes, ["state-dom.", "action-dom.", "ctx-dom."]):
        mid = cum + gs / 2
        ax.text(-0.6, mid, f"{label}\n(n={gs})", ha="right", va="center",
                fontsize=8, transform=ax.get_yaxis_transform())
        cum += gs

plt.tight_layout()
plt.savefig("figs/glm_neuron_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved to figs/glm_neuron_heatmap.png")

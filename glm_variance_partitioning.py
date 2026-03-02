import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from model import CerebellumForwardModelContinuous
from data import simulate_state_action_and_next_state


#Load model 

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(
    "checkpoints/cerebellum_forward_continuous_context_beta_gamma.pt",
    map_location=device,
)

model = CerebellumForwardModelContinuous(input_dim=6, hidden_dim=256).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Model loaded.")


# ── Generate data
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

S = states_xy       # (N, 2)  current state [x, y]
A = actions_vec     # (N, 2)  action [dx, dy]
C = contexts        # (N, 2)  context [beta, gamma]
Y = next_states_xy  # (N, 2)  ground truth next state [x', y']

print(f"Dataset size: {len(S)} samples")


# ── Compute context×action interaction features 
# beta*dx, beta*dy, gamma*dx, gamma*dy
CA = np.hstack([
    C[:, 0:1] * A[:, 0:1],   # beta  * dx
    C[:, 0:1] * A[:, 1:2],   # beta  * dy
    C[:, 1:2] * A[:, 0:1],   # gamma * dx
    C[:, 1:2] * A[:, 1:2],   # gamma * dy
])

# ── Define predictor sets ─────────────────────────────────────────────────────
# When dropping action  → also drop CA (interactions involve action)
# When dropping context → also drop CA (interactions involve context)

predictor_sets = {
    "Full (s+a+c+ca)":      np.hstack([S, A, C, CA]),
    "Drop state (a+c+ca)":  np.hstack([A, C, CA]),
    "Drop action (s+c)":    np.hstack([S, C]),
    "Drop context (s+a)":   np.hstack([S, A]),
    "Only state (s)":       S,
    "Only action (a)":      A,
    "Only context (c)":     C,
}


# ── Fit Ridge regression for each predictor set ───────────────────────────────
# Pipeline ensures StandardScaler is fit only on training folds (correct CV)

def ridge_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])


results = {}

print("\n--- R² scores (5-fold cross-validated) ---")
for name, X in predictor_sets.items():
    r2_x = cross_val_score(ridge_pipeline(), X, Y[:, 0], cv=5, scoring="r2").mean()
    r2_y = cross_val_score(ridge_pipeline(), X, Y[:, 1], cv=5, scoring="r2").mean()
    r2_mean = (r2_x + r2_y) / 2

    results[name] = {"x_next": r2_x, "y_next": r2_y, "mean": r2_mean}

    print(f"  {name}")
    print(f"    R² x_next: {r2_x:.4f}  |  R² y_next: {r2_y:.4f}  |  mean: {r2_mean:.4f}")


# ── Unique contributions: drop in R² from full model ─────────────────────────

full_r2 = results["Full (s+a+c+ca)"]["mean"]

print("\n--- Unique contribution of each variable (drop in mean R² from full) ---")
drop_labels = ["state", "action", "context"]
drop_keys   = ["Drop state (a+c+ca)", "Drop action (s+c)", "Drop context (s+a)"]
drop_vals   = [full_r2 - results[k]["mean"] for k in drop_keys]

for label, drop in zip(drop_labels, drop_vals):
    print(f"  {label}: {drop:.4f}")


#  Plot 

os.makedirs("figs", exist_ok=True)

names     = list(results.keys())
r2_x_vals = [results[n]["x_next"] for n in names]
r2_y_vals = [results[n]["y_next"] for n in names]
r2_mean_vals = [results[n]["mean"] for n in names]

x     = np.arange(len(names))
width = 0.35

fig, axes = plt.subplots(1, 3, figsize=(22, 5))

# Panel 1: R² for x_next and y_next separately
axes[0].bar(x - width / 2, r2_x_vals, width, label="x_next", color="steelblue")
axes[0].bar(x + width / 2, r2_y_vals, width, label="y_next", color="coral")
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
axes[0].set_ylabel("R²")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("R² per output dimension")
axes[0].legend()

# Panel 2: Mean R² across x and y
# full=blue, drop variants=grey, only variants=orange
colors = (
    ["steelblue"] +          # Full
    ["grey"] * 3 +           # Drop state, Drop action, Drop context
    ["darkorange"] * 3       # Only state, Only action, Only context
)
axes[1].bar(names, r2_mean_vals, color=colors)
axes[1].axhline(full_r2, color="red", linestyle="--", linewidth=1, label="Full model (s+a+c+ca)")
axes[1].set_xticks(range(len(names)))
axes[1].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
axes[1].set_ylabel("Mean R²")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Mean R² (averaged over x and y)")
axes[1].legend()

# Panel 3: Unique contributions (drop in R²)
bar_colors = ["steelblue", "coral", "green"]
axes[2].bar(drop_labels, drop_vals, color=bar_colors)
axes[2].set_ylabel("Drop in mean R² from full model")
axes[2].set_title("Unique contribution of each variable")
axes[2].set_ylim(0, max(drop_vals) * 1.3 if max(drop_vals) > 0 else 0.1)
for i, v in enumerate(drop_vals):
    axes[2].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("figs/glm_variance_partitioning_contextaction.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nSaved to figs/glm_variance_partitioning_contextaction.png")

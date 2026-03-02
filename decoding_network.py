# decoding_network.py
# Goal: can we linearly decode context (beta, gamma) from the network's
# hidden activations? Also decodes state and actions.

#import the packages needed for this 
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict

from model import CerebellumForwardModelContinuous
from data import simulate_state_action_and_next_state


# 1. Load the  model

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(
    "checkpoints/cerebellum_forward_continuous_context_beta_gamma.pt",
    map_location=device,
)

model = CerebellumForwardModelContinuous(input_dim=6, hidden_dim=256).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Model loaded.")


# 2. Generate new data (randomseed=123) 

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

S = states_xy       # (N, 2)  [x, y]
A = actions_vec     # (N, 2)  [dx, dy]
C = contexts        # (N, 2)  [beta, gamma]

print(f"Dataset size: {len(S)} samples")


# 3. Extract hidden activations (look at representations) via forward hooks
# we run a function that runs automatically every time a layer
# does its computation. Here we attach one to each ReLU layer to capture
# what the network's internal neurons are doing for each input sample.
#
# model.net is Sequential: [Linear, ReLU, Linear, ReLU, Linear]
#   index 1 = first ReLU  → layer 1 activations  (H1)
#   index 3 = second ReLU → layer 2 activations  (H2)

hidden_reps = {}

def make_hook(name):
    def hook(module, input, output):
        hidden_reps[name] = output.detach().cpu().numpy()
    return hook

hook_h1 = model.net[1].register_forward_hook(make_hook("h1"))
hook_h2 = model.net[3].register_forward_hook(make_hook("h2"))

with torch.no_grad():
    _ = model(
        torch.tensor(S, dtype=torch.float32).to(device),
        torch.tensor(A, dtype=torch.float32).to(device),
        torch.tensor(C, dtype=torch.float32).to(device),
    )

hook_h1.remove()
hook_h2.remove()

H1 = hidden_reps["h1"]   # (N, 256) activations after layer 1
H2 = hidden_reps["h2"]   # (N, 256) activations after layer 2

print(f"Hidden activations extracted: H1={H1.shape}, H2={H2.shape}")


# 4. Decode variables from hidden activations of layers
# We fit a simple Ridge regression from H2 → each target variable.
# If the network encodes a variable in its hidden layer, a linear decoder
# should be able to read it out (high R²).
# We decode context, state, and action to compare — state/action serve as
# an upper-bound reference since the network definitely uses them.

def ridge_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])

decode_targets = {
    "beta":  C[:, 0],
    "gamma": C[:, 1],
    "x":     S[:, 0],
    "y":     S[:, 1],
    "dx":    A[:, 0],
    "dy":    A[:, 1],
}

print("\n--- Decoding R² from layer 2 hidden activations ---")
r2_scores = {}
for name, target in decode_targets.items():
    r2 = cross_val_score(ridge_pipeline(), H2, target, cv=5, scoring="r2").mean()
    r2_scores[name] = r2
    print(f"  {name}: R² = {r2:.4f}")


# 5. Plot 

os.makedirs("figs", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Panel 1: Decoding R² bar chart
colors = ["green", "green", "steelblue", "steelblue", "coral", "coral"]
for i, (name, r2) in enumerate(r2_scores.items()):
    axes[0].bar(i, r2, color=colors[i])
    axes[0].text(i, r2 + 0.01, f"{r2:.2f}", ha="center", fontsize=9)

axes[0].set_xticks(range(len(r2_scores)))
axes[0].set_xticklabels(list(r2_scores.keys()))
axes[0].set_ylabel("Decoding R²")
axes[0].set_ylim(0, 1.1)
axes[0].set_title("Decoding variables from hidden activations (layer 2)")
axes[0].legend(handles=[
    Patch(facecolor="green",     label="context (beta, gamma)"),
    Patch(facecolor="steelblue", label="state (x, y)"),
    Patch(facecolor="coral",     label="action (dx, dy)"),
])

# Panel 2: Scatter of true vs decoded context values
beta_pred  = cross_val_predict(ridge_pipeline(), H2, C[:, 0], cv=5)
gamma_pred = cross_val_predict(ridge_pipeline(), H2, C[:, 1], cv=5)

axes[1].scatter(C[:, 0], beta_pred,  alpha=0.15, s=5, color="green",
                label=f"beta  (R²={r2_scores['beta']:.2f})")
axes[1].scatter(C[:, 1], gamma_pred, alpha=0.15, s=5, color="darkgreen",
                label=f"gamma (R²={r2_scores['gamma']:.2f})")
lims = [-0.6, 0.6]
axes[1].plot(lims, lims, "k--", linewidth=1, label="perfect decoding")
axes[1].set_xlabel("True context value")
axes[1].set_ylabel("Decoded from hidden activations")
axes[1].set_title("True vs decoded context")
axes[1].legend(markerscale=3)

plt.tight_layout()
plt.savefig("figs/decoding_network.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nSaved to figs/decoding_network.png")

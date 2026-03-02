# make_figs.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from data import simulate_state_action_and_next_state
from model import CerebellumForwardModelContinuous


def rmse_2d(y_true, y_pred):
    err = y_true - y_pred
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "checkpoints/cerebellum_forward_continuous_context_beta_gamma.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    model = CerebellumForwardModelContinuous(input_dim=6, hidden_dim=256).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    H, W = ckpt["H"], ckpt["W"]
    action_set = ckpt["action_set"]
    slip_prob = ckpt["slip_prob"]
    context_mask = tuple(ckpt["context_mask"])

    # Fresh eval dataset
    states, actions, next_states, contexts = simulate_state_action_and_next_state(
        n_runs=50, T=200, H=H, W=W,
        action_set=action_set, slip_prob=slip_prob,
        seed=123, #test on a random seed new dataset
        context_mask=context_mask,
        zero_disabled_in_context=True,
    )

    s = torch.tensor(states, dtype=torch.float32, device=device) #current state 
    a = torch.tensor(actions, dtype=torch.float32, device=device) #current action
    c = torch.tensor(contexts, dtype=torch.float32, device=device) #current context 
    y = torch.tensor(next_states, dtype=torch.float32, device=device) #next state true

    # Predictions under 3 context conditions
    with torch.no_grad():
        preds_true = model(s, a, c)

        c_zero = torch.zeros_like(c)
        preds_zero = model(s, a, c_zero)

        perm = torch.randperm(c.shape[0], device=device)
        preds_shuf = model(s, a, c[perm])

    # Move to numpy for plotting
    y_np = y.cpu().numpy() #true next state 
    p_true = preds_true.cpu().numpy() #predicted next state w correct contexts 
    p_zero = preds_zero.cpu().numpy() #predicted next state w no context info 
    p_shuf = preds_shuf.cpu().numpy() #predicted next state w shuffled context info 
    c_np = c.cpu().numpy()

    os.makedirs("figs", exist_ok=True)

    #Figures 
    # Fig 1: plot predictions vs real x and y

    for dim, name in [(0, "x_next"), (1, "y_next")]:
        plt.figure()
        plt.scatter(y_np[:, dim], p_true[:, dim], s=6)
        lo = float(min(y_np[:, dim].min(), p_true[:, dim].min()))
        hi = float(max(y_np[:, dim].max(), p_true[:, dim].max()))
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Pred {name}")
        plt.title(f"Predicted vs True: {name}")
        plt.tight_layout()
        plt.savefig(f"figs/fig1_pred_vs_true_{name}.png", dpi=200)
        plt.close()

    
    # Fig 2: Error histogram (Euclidean per-sample error)

    err_true = np.linalg.norm(y_np - p_true, axis=1)
    plt.figure()
    plt.hist(err_true, bins=50)
    plt.xlabel("||s_next - s_pred||")
    plt.ylabel("Count")
    plt.title("One-step prediction error distribution")
    plt.tight_layout()
    plt.savefig("figs/fig2_error_hist.png", dpi=200)
    plt.close()

    # Fig 3: Context ablation (RMSE bars) -- to know how much does context play a role in the model predictions 
    # if i remove a context signal, how does prediction get affected  
    
    rmse_true = rmse_2d(y_np, p_true)
    rmse_zero = rmse_2d(y_np, p_zero)
    rmse_shuf = rmse_2d(y_np, p_shuf)

    plt.figure()
    labels = ["True context", "Zero context", "Shuffled context"]
    vals = [rmse_true, rmse_zero, rmse_shuf]
    plt.bar(labels, vals)
    plt.ylabel("RMSE_2D")
    plt.title("Context ablation: does context matter?")
    plt.tight_layout()
    plt.savefig("figs/fig3_context_ablation.png", dpi=200)
    plt.close()

    # -----------------------
    # Fig 4: Sensitivity sweep (beta and gamma effects)
    # -----------------------
    state0 = torch.tensor([[2.0, 2.0]], device=device)
    action0 = torch.tensor([[1.0, 0.0]], device=device)  # fixed nonzero action
    betas = torch.linspace(-0.5, 0.5, 101, device=device)
    gammas = torch.linspace(-0.5, 0.5, 101, device=device)

    # Sweep beta with gamma=0
    with torch.no_grad():
        c_beta = torch.stack([betas, torch.zeros_like(betas)], dim=1)
        s_rep = state0.repeat(betas.shape[0], 1)
        a_rep = action0.repeat(betas.shape[0], 1)
        pred_beta = model(s_rep, a_rep, c_beta).cpu().numpy()

    # Sweep gamma with beta=0
    with torch.no_grad():
        c_gamma = torch.stack([torch.zeros_like(gammas), gammas], dim=1)
        s_rep = state0.repeat(gammas.shape[0], 1)
        a_rep = action0.repeat(gammas.shape[0], 1)
        pred_gamma = model(s_rep, a_rep, c_gamma).cpu().numpy()

    # Plot predicted deltas (predicted next state - current state)
    state0_np = state0.cpu().numpy()[0]

    plt.figure()
    plt.plot(betas.cpu().numpy(), pred_beta[:, 0] - state0_np[0], label="Δx")
    plt.plot(betas.cpu().numpy(), pred_beta[:, 1] - state0_np[1], label="Δy")
    plt.xlabel("beta (gamma=0)")
    plt.ylabel("Predicted delta")
    plt.title("Sensitivity to beta (should align with action)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/fig4a_beta_sensitivity.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(gammas.cpu().numpy(), pred_gamma[:, 0] - state0_np[0], label="Δx")
    plt.plot(gammas.cpu().numpy(), pred_gamma[:, 1] - state0_np[1], label="Δy")
    plt.xlabel("gamma (beta=0)")
    plt.ylabel("Predicted delta")
    plt.title("Sensitivity to gamma (should be perpendicular)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/fig4b_gamma_sensitivity.png", dpi=200)
    plt.close()

    # -----------------------
    # Fig 5: Error vs context magnitude (true vs zeroed context)
    # -----------------------
    ctx_mag = np.sqrt(c_np[:, 0] ** 2 + c_np[:, 1] ** 2)
    err_zero = np.linalg.norm(y_np - p_zero, axis=1)

    plt.figure()
    plt.scatter(ctx_mag, err_true, s=6, label="True context")
    plt.scatter(ctx_mag, err_zero, s=6, label="Zeroed context")
    plt.xlabel("||context|| = sqrt(beta^2 + gamma^2)")
    plt.ylabel("||prediction error||")
    plt.title("Error vs context magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/fig5_error_vs_context_mag.png", dpi=200)
    plt.close()

    print("Saved figures to ./figs/")
    print("RMSE_2D true / zero / shuffled:", rmse_true, rmse_zero, rmse_shuf)


if __name__ == "__main__":
    main()

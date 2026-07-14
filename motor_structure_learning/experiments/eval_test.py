import numpy as np
import torch

from motor_structure_learning.core import (
    CerebellumForwardModelContinuous,
    simulate_state_action_and_next_state,
)


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

    states, actions, next_states, contexts = simulate_state_action_and_next_state(
        n_runs=50,
        T=200,
        H=H,
        W=W,
        action_set=action_set,
        slip_prob=slip_prob,
        seed=123,
        context_mask=context_mask,
        zero_disabled_in_context=True,
    )

    s = torch.tensor(states, dtype=torch.float32, device=device)
    a = torch.tensor(actions, dtype=torch.float32, device=device)
    c = torch.tensor(contexts, dtype=torch.float32, device=device)
    y = torch.tensor(next_states, dtype=torch.float32, device=device)

    with torch.no_grad():
        preds = model(s, a, c)
        mse = torch.mean((preds - y) ** 2).item()
        err = (y - preds).cpu().numpy()
        rmse_2d = np.sqrt(np.mean(np.sum(err ** 2, axis=1)))

    print("Eval MSE:", mse)
    print("Eval RMSE_2D:", rmse_2d)


if __name__ == "__main__":
    main()
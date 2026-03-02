# train.py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from data import simulate_state_action_and_next_state
from model import CerebellumForwardModelContinuous


class ForwardDatasetContinuous(Dataset):
    def __init__(self, states_xy, actions_vec, contexts, next_states_xy):
        self.states = torch.tensor(states_xy, dtype=torch.float32)
        self.actions = torch.tensor(actions_vec, dtype=torch.float32)
        self.contexts = torch.tensor(contexts, dtype=torch.float32)  # (beta, gamma)
        self.next_states = torch.tensor(next_states_xy, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.contexts[idx], self.next_states[idx]


def train():
    H, W = 5, 5
    action_set = "4"
    slip_prob = 0.0

    batch_size = 128
    epochs = 20
    lr = 1e-3

    # Toggle contexts here: (beta, gamma)
    # (1,1) -> along + perpendicular forces
    # (1,0) -> along-only
    # (0,1) -> perpendicular-only (curl-like)
    # (0,0) -> no force field; pure action dynamics
    context_mask = (1, 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Context mask (beta,gamma):", context_mask)

    states_xy, actions_vec, next_states_xy, contexts = simulate_state_action_and_next_state(
        n_runs=300,
        T=200,
        H=H,
        W=W,
        action_set=action_set,
        slip_prob=slip_prob,
        seed=0,
        beta_prob=0.7,
        beta_range=(-0.5, 0.5),
        gamma_prob=0.7,
        gamma_range=(-0.5, 0.5),
        context_mask=context_mask,
        zero_disabled_in_context=True,
    )

    s_tr, s_va, a_tr, a_va, c_tr, c_va, y_tr, y_va = train_test_split(
        states_xy, actions_vec, contexts, next_states_xy,
        test_size=0.2, random_state=0
    )

    train_ds = ForwardDatasetContinuous(s_tr, a_tr, c_tr, y_tr)
    val_ds = ForwardDatasetContinuous(s_va, a_va, c_va, y_va)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # input_dim = 2(state) + 2(action) + 2(context) = 6
    model = CerebellumForwardModelContinuous(input_dim=6, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for s, a, c, y in train_loader:
            s, a, c, y = s.to(device), a.to(device), c.to(device), y.to(device)
            preds = model(s, a, c)
            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * s.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        preds_all, y_all = [], []

        with torch.no_grad():
            for s, a, c, y in val_loader:
                s, a, c, y = s.to(device), a.to(device), c.to(device), y.to(device)
                preds = model(s, a, c)
                val_loss += loss_fn(preds, y).item() * s.size(0)
                preds_all.append(preds.cpu().numpy())
                y_all.append(y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        preds_all = np.concatenate(preds_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        err = y_all - preds_all
        rmse_2d = np.sqrt(np.mean(np.sum(err ** 2, axis=1)))

        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f} | "
            f"RMSE_2D: {rmse_2d:.4f}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/cerebellum_forward_continuous_context_beta_gamma.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "H": H,
            "W": W,
            "action_set": action_set,
            "slip_prob": slip_prob,
            "context_mask": context_mask,
        },
        ckpt_path,
    )
    print(f"Saved model to {ckpt_path}")

    return model


if __name__ == "__main__":
    train()

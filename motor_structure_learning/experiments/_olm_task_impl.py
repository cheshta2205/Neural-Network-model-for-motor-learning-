"""
Object Location Memory (OLM) task.

Extends the EC -> CB prediction-error-minimization task (see ECtoCB_MSE.py,
train_shifted()) with two static objects in the environment. EC and CB are
trained exactly the same way as the base task -- minimize MSE prediction
error of the next state under a random-action trajectory -- except the state
now includes egocentric perceptual features of both objects, not just agent
position.

Objects never move during training: the network only ever learns "objects
are static." Displacement is introduced solely at evaluation time
(evaluate_displacement), as a held-out probe -- so a spike in CB's
prediction error for the relocated object's features is a genuine,
untrained novelty-detection signal, analogous to preferential exploration in
real object-location-memory experiments.

Actions are continuous: a random direction (angle) and a random magnitude in
[0, MAX_STEP], applied to agent position bounded within [-POS_BOUND, POS_BOUND]
per axis. Heading is the sampled direction itself (not the resulting
displacement), so it stays well-defined even when the sampled magnitude is
near zero.

Flow (per timestep):
  agent state    = [x, y]
  object feature = [identity, normalized_distance, sin(bearing), cos(bearing)]
                   (bearing is egocentric: relative to the agent's heading,
                   defined as the direction of the last action taken)
  state          = [agent state, object_0 feature, object_1 feature]
  [state_t, action_t, state_{t+1}] -> EC (GRU) -> z_t
  [state_{t+1}, action_{t+1}, z_t] -> CB (MLP) -> predicted state_{t+2}
                                            |
                          MSE(predicted, actual state_{t+2})
                                            |
                          gradient flows back through CB -> EC
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from motor_structure_learning.core import EntorhinalCortex
from src.PlottingTools.plotting_functions import make_axis_nice, set_plt_params

# Hyperparameters
POS_BOUND = 5.0   # agent/object positions are bounded in [-POS_BOUND, POS_BOUND] per axis
MAX_STEP = 1.0    # max action magnitude (direction and magnitude are both drawn continuously)
SEQ_LEN = 60
N_TRAIN = 2000
N_VAL = 200
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
LATENT_DIM = 32
CB_HIDDEN = 256

N_OBJECTS = 2
IDENTITY_DIM = 8

ACTION_DIM = 2
OBJ_FEATURE_DIM = IDENTITY_DIM + 3          # identity + distance + sin + cos
STATE_DIM = 2 + N_OBJECTS * OBJ_FEATURE_DIM  # agent xy + both object slots
TRANS_DIM = 2 * STATE_DIM + ACTION_DIM       # [state_t, action_t, state_{t+1}]

CKPT_PATH = "checkpoints/olm_task.pt"
FIG_PATH = "figs/olm_displacement_probe.png"
MULTI_SEED_DIR = "checkpoints/olm_multi_seed"


def _object_feature(agent_pos, heading, obj_pos, identity, diag):
    """Egocentric feature for one object: identity + distance/bearing relative to agent heading."""
    rel = obj_pos - agent_pos
    dist = float(np.linalg.norm(rel)) / diag
    phi = float(np.arctan2(rel[1], rel[0]))
    theta_h = float(np.arctan2(heading[1], heading[0]))
    theta = phi - theta_h
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return np.concatenate([identity, [dist, np.sin(theta), np.cos(theta)]]).astype(np.float32)


def _build_state(agent_pos, heading, obj_positions, identities, diag):
    obj_feats = [
        _object_feature(agent_pos, heading, obj_positions[i], identities[i], diag)
        for i in range(len(identities))
    ]
    return np.concatenate([agent_pos.astype(np.float32)] + obj_feats)


def _sample_actions(rng, n, max_step=MAX_STEP):
    """n continuous random actions: random direction (angle) and random magnitude in [0, max_step].

    Heading is defined by the sampled direction alone (not the resulting displacement), so it
    stays well-defined even for near-zero-magnitude actions -- no normalization/fallback needed.
    """
    angles = rng.uniform(0.0, 2 * np.pi, size=n).astype(np.float32)
    magnitudes = rng.uniform(0.0, max_step, size=n).astype(np.float32)
    headings = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
    actions = headings * magnitudes[:, None]
    return actions, headings


def generate_olm_sequences(n_seqs, seq_len=SEQ_LEN, pos_bound=POS_BOUND, max_step=MAX_STEP,
                            n_objects=N_OBJECTS, identity_dim=IDENTITY_DIM, seed=0):
    """Random-action trajectories with two static objects (fresh identity/location per sequence)."""
    rng = np.random.default_rng(seed)
    diag = float(2 * pos_bound * np.sqrt(2))

    all_trans = []
    for _ in range(n_seqs):
        agent_pos = rng.uniform(-pos_bound, pos_bound, size=2).astype(np.float32)
        identities = rng.standard_normal((n_objects, identity_dim)).astype(np.float32)
        obj_positions = rng.uniform(-pos_bound, pos_bound, size=(n_objects, 2)).astype(np.float32)
        heading = np.array([1.0, 0.0], dtype=np.float32)

        actions, headings = _sample_actions(rng, seq_len, max_step)

        rows = []
        for t in range(seq_len):
            state_t = _build_state(agent_pos, heading, obj_positions, identities, diag)

            action = actions[t]
            next_pos = np.clip(agent_pos + action, -pos_bound, pos_bound)
            next_heading = headings[t]

            state_next = _build_state(next_pos, next_heading, obj_positions, identities, diag)
            rows.append(np.concatenate([state_t, action, state_next]))

            agent_pos, heading = next_pos, next_heading

        all_trans.append(np.stack(rows))

    return torch.tensor(np.stack(all_trans), dtype=torch.float32)  # (N, T, TRANS_DIM)


def generate_displacement_probe_data(n_pairs, seq_len=SEQ_LEN, t_probe=None, pos_bound=POS_BOUND,
                                      max_step=MAX_STEP, n_objects=N_OBJECTS, identity_dim=IDENTITY_DIM, seed=0):
    """Paired control/displaced episodes sharing the same trajectory + objects, up to t_probe.

    In the displaced variant, one randomly chosen object is relocated to a new random
    position at t_probe; both variants are otherwise identical (same agent trajectory,
    same initial object identities/locations).
    """
    if t_probe is None:
        t_probe = seq_len // 2
    rng = np.random.default_rng(seed)
    diag = float(2 * pos_bound * np.sqrt(2))

    control_trans, displaced_trans, moved_idxs = [], [], []

    for _ in range(n_pairs):
        start_pos = rng.uniform(-pos_bound, pos_bound, size=2).astype(np.float32)
        identities = rng.standard_normal((n_objects, identity_dim)).astype(np.float32)
        obj_positions0 = rng.uniform(-pos_bound, pos_bound, size=(n_objects, 2)).astype(np.float32)
        actions, headings = _sample_actions(rng, seq_len, max_step)  # shared trajectory

        moved_idx = int(rng.integers(0, n_objects))
        new_loc = rng.uniform(-pos_bound, pos_bound, size=2).astype(np.float32)

        for displaced, trans_list in ((False, control_trans), (True, displaced_trans)):
            agent_pos = start_pos.copy()
            heading = np.array([1.0, 0.0], dtype=np.float32)
            obj_positions = obj_positions0.copy()

            rows = []
            for t in range(seq_len):
                state_t = _build_state(agent_pos, heading, obj_positions, identities, diag)

                action = actions[t]
                next_pos = np.clip(agent_pos + action, -pos_bound, pos_bound)
                next_heading = headings[t]

                if displaced and (t + 1) == t_probe:
                    obj_positions = obj_positions.copy()
                    obj_positions[moved_idx] = new_loc

                state_next = _build_state(next_pos, next_heading, obj_positions, identities, diag)
                rows.append(np.concatenate([state_t, action, state_next]))

                agent_pos, heading = next_pos, next_heading

            trans_list.append(np.stack(rows))

        moved_idxs.append(moved_idx)

    return (
        torch.tensor(np.stack(control_trans), dtype=torch.float32),
        torch.tensor(np.stack(displaced_trans), dtype=torch.float32),
        np.array(moved_idxs),
        t_probe,
    )


class OLMForwardModel(nn.Module):
    """MLP forward model: [state, action, z] -> predicted next state. Mirrors CerebellumCB
    (ECtoCB_MSE.py) but with a configurable output dim, since the shared
    CerebellumForwardModelContinuous hardcodes a 2-dim output."""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM, hidden=CB_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, s, a, z):
        return self.net(torch.cat([s, a, z], dim=-1))


def _shifted_forward(ec, cb, trans):
    """z_t predicts one step ahead, mirroring train_shifted() in ECtoCB_MSE.py."""
    z, _ = ec(trans)
    z_prev = z[:, :-1, :]
    s_curr = trans[:, 1:, :STATE_DIM]
    a_curr = trans[:, 1:, STATE_DIM:STATE_DIM + ACTION_DIM]
    s_target = trans[:, 1:, STATE_DIM + ACTION_DIM:]
    s_pred = cb(s_curr, a_curr, z_prev)
    return s_pred, s_target


def train_olm(n_train=N_TRAIN, n_val=N_VAL, seq_len=SEQ_LEN, epochs=EPOCHS,
              batch_size=BATCH_SIZE, lr=LR, seed=0, device=None, ckpt_path=CKPT_PATH):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training sequences: {n_train} | Val: {n_val} | Epochs: {epochs}")

    tr_trans = generate_olm_sequences(n_train, seq_len=seq_len, seed=seed)
    va_trans = generate_olm_sequences(n_val, seq_len=seq_len, seed=seed + 1000)

    train_loader = DataLoader(TensorDataset(tr_trans), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(va_trans), batch_size=batch_size)

    ec = EntorhinalCortex(input_dim=TRANS_DIM, hidden_dim=256, latent_dim=LATENT_DIM).to(device)
    cb = OLMForwardModel(state_dim=STATE_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM, hidden=CB_HIDDEN).to(device)
    optimizer = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=lr)
    history = {"train_mse": [], "val_mse": []}

    print("\nTraining EC -> CB on the OLM task (MSE prediction loss)...\n")

    for epoch in range(1, epochs + 1):
        ec.train(); cb.train()
        total_mse = 0.0
        for (trans,) in train_loader:
            trans = trans.to(device)
            s_pred, s_target = _shifted_forward(ec, cb, trans)
            loss = F.mse_loss(s_pred, s_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mse += loss.item() * trans.shape[0]
        total_mse /= n_train
        history["train_mse"].append(total_mse)

        ec.eval(); cb.eval()
        val_mse = 0.0
        with torch.no_grad():
            for (trans,) in val_loader:
                trans = trans.to(device)
                s_pred, s_target = _shifted_forward(ec, cb, trans)
                val_mse += F.mse_loss(s_pred, s_target).item() * trans.shape[0]
        val_mse /= n_val
        history["val_mse"].append(val_mse)
        print(f"Epoch {epoch:03d} | Train MSE: {total_mse:.4f} | Val MSE: {val_mse:.4f}")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "history": history}, ckpt_path)
    print(f"\nSaved {ckpt_path}")

    return ec, cb, history, device


def _slot_error(s_pred, s_target, obj_idx):
    start = 2 + obj_idx * OBJ_FEATURE_DIM
    end = start + OBJ_FEATURE_DIM
    return ((s_pred[..., start:end] - s_target[..., start:end]) ** 2).mean(dim=-1)  # (B, T-1)


def evaluate_displacement(ec, cb, n_pairs=200, seq_len=SEQ_LEN, t_probe=None, seed=1234,
                           device=None, fig_path=FIG_PATH):
    """Held-out displacement probe: compare per-object prediction error between
    control (objects static) and displaced (one object relocated at t_probe) episodes."""
    device = device or next(ec.parameters()).device
    ec.eval(); cb.eval()

    control_trans, displaced_trans, moved_idxs, t_probe = generate_displacement_probe_data(
        n_pairs, seq_len=seq_len, t_probe=t_probe, seed=seed,
    )

    with torch.no_grad():
        s_pred_c, s_target_c = _shifted_forward(ec, cb, control_trans.to(device))
        s_pred_d, s_target_d = _shifted_forward(ec, cb, displaced_trans.to(device))

    other_idxs = 1 - moved_idxs  # only valid for N_OBJECTS == 2

    def gather(s_pred, s_target, idxs):
        return torch.stack([
            _slot_error(s_pred[i], s_target[i], int(idxs[i])) for i in range(len(idxs))
        ])

    err_moved_disp = gather(s_pred_d, s_target_d, moved_idxs)
    err_moved_ctrl = gather(s_pred_c, s_target_c, moved_idxs)
    err_other_disp = gather(s_pred_d, s_target_d, other_idxs)

    mean_moved_disp = err_moved_disp.mean(dim=0).cpu().numpy()
    mean_moved_ctrl = err_moved_ctrl.mean(dim=0).cpu().numpy()
    mean_other_disp = err_other_disp.mean(dim=0).cpu().numpy()

    std_moved_disp = err_moved_disp.std(dim=0).cpu().numpy()
    std_moved_ctrl = err_moved_ctrl.std(dim=0).cpu().numpy()
    std_other_disp = err_other_disp.std(dim=0).cpu().numpy()

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    # Selection index j (0-indexed into s_pred/s_target) predicts state(j+2):
    # z_prev[j] has seen transitions through row j (i.e. through state(j+1)),
    # and s_target[j] is state_next of row (j+1), i.e. state(j+2).
    t_axis = np.arange(2, seq_len + 1)
    set_plt_params(style="fast")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_axis, mean_moved_disp, label="moved object, displaced episode", color="tomato")
    ax.fill_between(t_axis, mean_moved_disp - std_moved_disp / np.sqrt(len(err_moved_disp)),
                     mean_moved_disp + std_moved_disp / np.sqrt(len(err_moved_disp)), color="tomato", alpha=0.25)

    ax.plot(t_axis, mean_moved_ctrl, label="same object, control episode", color="steelblue")
    ax.fill_between(t_axis, mean_moved_ctrl - std_moved_ctrl / np.sqrt(len(err_moved_ctrl)),
                     mean_moved_ctrl + std_moved_ctrl / np.sqrt(len(err_moved_ctrl)), color="steelblue", alpha=0.25)

    ax.plot(t_axis, mean_other_disp, label="unmoved object, displaced episode", color="seagreen")
    ax.fill_between(t_axis, mean_other_disp - std_other_disp / np.sqrt(len(err_other_disp)),
                     mean_other_disp + std_other_disp / np.sqrt(len(err_other_disp)), color="seagreen", alpha=0.25)
    
    ax.axvline(t_probe, color="gray", linestyle="--", label=f"relocation (t={t_probe})")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Mean squared prediction error")
    ax.set_title("OLM displacement probe")
    ax.legend()

    make_axis_nice(ax)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    return {
        "mean_moved_displaced": mean_moved_disp,
        "mean_moved_control": mean_moved_ctrl,
        "mean_other_displaced": mean_other_disp,
        "std_moved_displaced": std_moved_disp,
        "std_moved_control": std_moved_ctrl,
        "std_other_displaced": std_other_disp,
        "t_probe": t_probe,
    }


def _save_training_history(seed_dir, history):
    """Save per-epoch train/val MSE separately from the checkpoint (history.npz + a quick plot),
    so loss curves can be inspected/aggregated without loading model weights."""
    np.savez(
        os.path.join(seed_dir, "training_history.npz"),
        train_mse=np.array(history["train_mse"]),
        val_mse=np.array(history["val_mse"]),
    )

    epochs = np.arange(1, len(history["train_mse"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_mse"], label="train MSE", color="steelblue")
    ax.plot(epochs, history["val_mse"], label="val MSE", color="tomato")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title("OLM training curve")
    ax.legend()
    make_axis_nice(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(seed_dir, "training_curve.png"), bbox_inches="tight")
    plt.close(fig)


def train_olm_multi_seed(seeds, out_dir=MULTI_SEED_DIR, train_kwargs=None, eval_kwargs=None):
    """Train + evaluate one OLM model per seed, saving each under out_dir/seed_{seed}/:
    model.pt (ec/cb/history), training_history.npz + training_curve.png,
    displacement_probe.png, displacement_probe_results.npz.

    Both train_olm and evaluate_displacement are called with the same `seed`, so a
    given seed_dir is fully determined by that one seed. Pass shared overrides (e.g.
    seq_len, epochs) via train_kwargs/eval_kwargs -- these are applied to every seed.

    Returns a list of {"seed", "seed_dir", "history", "probe"} dicts, one per seed.
    """
    train_kwargs = dict(train_kwargs or {})
    eval_kwargs = dict(eval_kwargs or {})

    results = []
    for seed in seeds:
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        print(f"\n{'=' * 60}\nOLM multi-seed: seed={seed}\n{'=' * 60}")
        ec, cb, history, device = train_olm(
            seed=seed, ckpt_path=os.path.join(seed_dir, "model.pt"), **train_kwargs,
        )
        _save_training_history(seed_dir, history)

        probe = evaluate_displacement(
            ec, cb, seed=seed, device=device,
            fig_path=os.path.join(seed_dir, "displacement_probe.png"), **eval_kwargs,
        )
        np.savez(os.path.join(seed_dir, "displacement_probe_results.npz"), **probe)

        results.append({"seed": seed, "seed_dir": seed_dir, "history": history, "probe": probe})

    return results


def main(seeds=None, out_dir=MULTI_SEED_DIR, train_kwargs=None, eval_kwargs=None):
    """Default: single-seed train_olm() + evaluate_displacement(), as before.

    Pass `seeds` (e.g. a list of ints) to switch to train_olm_multi_seed instead --
    trains + evaluates one model per seed, saved under out_dir/seed_{seed}/.
    """
    set_plt_params(style="fast", file_format="png")

    if seeds is not None:
        if isinstance(seeds, int):
            seeds = np.arange(seeds)

        return train_olm_multi_seed(seeds, out_dir=out_dir, train_kwargs=train_kwargs, eval_kwargs=eval_kwargs)

    ec, cb, history, device = train_olm()
    evaluate_displacement(ec, cb, device=device)


if __name__ == "__main__":
    main()

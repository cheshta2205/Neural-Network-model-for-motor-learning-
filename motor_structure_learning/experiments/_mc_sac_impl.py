"""Motor-learning SAC experiment.

This is a CleanRL-style off-policy SAC loop for the current motor task.
It keeps the implementation flat:

- actor / critic networks live here
- replay buffer lives here
- the trainer owns SAC update logic
- rollout collection and logging stay in the experiment loop
"""

from __future__ import annotations

import math
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from motor_structure_learning.core.trainers import SACBatch, SACTrainer

from src.PlottingTools.plotting_functions import make_axis_nice, set_plt_params


OBS_DIM = 6
ACTION_DIM = 2
HIDDEN_DIM = 256

MAX_STEPS = 100
N_PHASE_EP = 1000
GOAL_RADIUS = 0.25
GRID_SIZE = 4.0
DT = 1.0

LR_ACTOR = 3e-4
LR_Q = 3e-4
GAMMA = 0.99
TAU = 0.005
BUFFER_CAPACITY = 10000
BATCH_SIZE = 256
LEARNING_STARTS = 5000
UPDATE_EVERY = 1
GRAD_CLIP_NORM = 1.0

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

SUCCESS_BONUS = 5.0
FIG_DIR = "figs/mc_sac"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CTX_A = (1.0, np.pi / 3, 0.0, 0.0, 0.0, 0.0, "A: rot+60") # (gain, rotation, curl, drift_strength, drift_x, drift_y, label)
CTX_B = (1.0, 0.0, +0.4, 0.0, 0.0, 0.0, "B: curl+")

PHASE_REPS = 3

PHASE_DATA = [
    {"label": "A (rot +60°)", "params": CTX_A},
    {"label": "B (curl+)", "params": CTX_B}
]

N_PHASES = len(PHASE_DATA) * PHASE_REPS

def env_step_np(pos, vel, action, beta, theta, gamma_curl, g_dyn, bx, by):
    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s], [s, c]], dtype=np.float32)
    curl = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    b_v = np.array([bx, by], dtype=np.float32)
    f_env = beta * (r @ action) + gamma_curl * (curl @ vel) + g_dyn * b_v
    vel_new = vel + f_env
    pos_new = np.clip(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new


def build_obs(pos, goal, prev_action):
    return np.concatenate([pos, goal, prev_action], axis=0).astype(np.float32)


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.capacity = capacity
        self.observations = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_observations = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def __len__(self):
        return len(self.observations)

    def add(self, obs, action, reward, next_obs, done):
        self.observations.append(np.asarray(obs, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(np.asarray([reward], dtype=np.float32))
        self.next_observations.append(np.asarray(next_obs, dtype=np.float32).copy())
        self.dones.append(np.asarray([done], dtype=np.float32))

    def sample(self, batch_size, device):
        idxs = np.random.randint(0, len(self.observations), size=batch_size)
        observations = torch.tensor(np.stack([self.observations[i] for i in idxs]), device=device)
        actions = torch.tensor(np.stack([self.actions[i] for i in idxs]), device=device)
        rewards = torch.tensor(np.stack([self.rewards[i] for i in idxs]), device=device)
        next_observations = torch.tensor(np.stack([self.next_observations[i] for i in idxs]), device=device)
        dones = torch.tensor(np.stack([self.dones[i] for i in idxs]), device=device)
        return SACBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
        )


class Actor(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(HIDDEN_DIM, action_dim)
        self.fc_logstd = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, obs):
        hidden = self.backbone(obs)
        mean = self.fc_mean(hidden)
        log_std = self.fc_logstd(hidden)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return y_t, log_prob, mean


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


def phase_boundaries():
    return [i * N_PHASE_EP for i in range(1, len(PHASES))]


def add_phase_labels(ax):
    for xb in phase_boundaries():
        ax.axvline(xb, color="gray", ls="--", lw=0.8)
    for i, lbl in enumerate(PHASE_LABELS):
        ax.text((i + 0.5) * N_PHASE_EP, 1.02, lbl, ha="center", transform=ax.get_xaxis_transform(), fontsize=7)


def smooth(x, w=25):
    x = np.asarray(x)
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x


def plot_seed(seed, curve, radius, fig_dir, key_params=None):
    os.makedirs(fig_dir, exist_ok=True)
    ep_g = np.arange(1, len(curve["success"]) + 1)

    set_plt_params(style='fast')

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    
    for ax, key, color, label in [
        (axes[0], "q_loss", "tomato", "Q loss"),
        (axes[1], "actor_loss", "steelblue", "Actor loss"),
        (axes[2], "alpha", "darkorange", "Alpha"),
    ]:
        y = smooth(curve[key])
        ax.plot(ep_g[24:], y, color=color, lw=1.5)
        ax.axhline(0, color="black", ls=":", lw=0.7)
        ax.set_ylabel(label, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].set_xlabel("Episode (global)")
    fig.suptitle(f"SAC diagnostic curves (radius {radius}, seed {seed})", fontsize=11)
    make_axis_nice(fig)
    plt.savefig(os.path.join(fig_dir, "diagnostic_losses"))
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    y = smooth(curve["success"])
    ax.plot(ep_g[24:], y, lw=1.7)

    ax.set_ylim(0, 1)
    ax.set_xlim(ep_g[0], ep_g[-1] + 1)
    ax.set_ylabel("Success")
    ax.set_xlabel("Episode (global)")
    fig.suptitle(f"Episode success curve\n (radius {radius}, seed {seed})", fontsize=11)
    if key_params is not None:
        param_str = ", ".join(f"{k}={v}" for k, v in key_params.items())
        ax.set_title(param_str, fontsize=7)
    
    ax.spines[["top", "right"]].set_visible(False)
    make_axis_nice(ax)
    plt.savefig(os.path.join(fig_dir, "success_curve"))
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    y = smooth(curve["return"])
    ax.plot(ep_g[24:], y, lw=1.7, color="seagreen")
    # add_phase_labels(ax)
    ax.set_ylabel("Return")
    ax.set_ylim(-40, SUCCESS_BONUS)
    ax.set_xlim(ep_g[0], ep_g[-1] + 1)
    ax.set_xlabel("Episode (global)")
    fig.suptitle(f"Episode return curve\n (radius {radius}, seed {seed})", fontsize=11)
    if key_params is not None:
        param_str = ", ".join(f"{k}={v}" for k, v in key_params.items())
        ax.set_title(param_str, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    make_axis_nice(ax)
    plt.savefig(os.path.join(fig_dir, "return_curve"))
    plt.close()


def run_babab(seed=0):
    device = DEVICE
    rng = np.random.default_rng(seed)

    actor = Actor().to(device)
    qf1 = SoftQNetwork().to(device)
    qf2 = SoftQNetwork().to(device)
    qf1_target = SoftQNetwork().to(device)
    qf2_target = SoftQNetwork().to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=LR_Q)

    trainer = SACTrainer(
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        alpha=0.2,
        gamma=GAMMA,
        tau=TAU,
        target_entropy=-float(ACTION_DIM),
        autotune=True,
        device=device,
        grad_clip_norm=GRAD_CLIP_NORM,
    )

    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    curve = {
        "ep": [],
        "phase": [],
        "success": [],
        "steps": [],
        "dist_norm": [],
        "angle": [],
        "first_angle": [],
        "return": [],
        "q_loss": [],
        "actor_loss": [],
        "alpha": [],
        "q_target_mean": [],
        "log_pi_mean": [],
    }

    global_step = 0
    global_ep = 0

    for phase_idx, ctx in enumerate(PHASES):
        print(f"\nPhase {PHASE_LABELS[phase_idx]} ({N_PHASE_EP} episodes)...")

        for local_ep in range(1, N_PHASE_EP + 1):
            global_ep += 1

            beta, theta, gamma_curl, g_dyn, bx, by, _ = ctx
            pos_np = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
            goal_np = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
            while np.linalg.norm(pos_np - goal_np) < 0.5:
                goal_np = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

            vel_np = np.zeros(2, dtype=np.float32)
            prev_action_np = np.zeros(2, dtype=np.float32)
            obs_np = build_obs(pos_np, goal_np, prev_action_np)
            init_dist = float(np.linalg.norm(pos_np - goal_np))

            rewards = []
            dist_seq = []
            angle_seq = []
            first_angle = None
            success = False
            steps = 0
            latest_update = {"q_loss": 0.0, "actor_loss": 0.0, "alpha": trainer.current_alpha, "q_target_mean": 0.0, "log_pi_mean": 0.0}

            for _ in range(MAX_STEPS):
                d = float(np.linalg.norm(pos_np - goal_np))
                dist_seq.append(d)
                if d < GOAL_RADIUS:
                    success = True
                    break

                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                if global_step < LEARNING_STARTS or len(replay_buffer) < BATCH_SIZE:
                    action_np = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
                else:
                    with torch.no_grad():
                        action_t, _, _ = trainer.act(obs_t)
                    action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)

                s_next_np, vel_new_np = env_step_np(pos_np, vel_np, action_np, beta, theta, gamma_curl, g_dyn, bx, by)
                done = bool(np.linalg.norm(s_next_np - goal_np) < GOAL_RADIUS)
                reward = -0.1 * float(np.linalg.norm(s_next_np - goal_np))
                if done:
                    reward += SUCCESS_BONUS
                rewards.append(reward)

                gd = goal_np - pos_np
                cross = gd[0] * action_np[1] - gd[1] * action_np[0]
                dot = action_np[0] * gd[0] + action_np[1] * gd[1]
                alpha = float(np.degrees(np.arctan2(cross, dot)))
                angle_seq.append(alpha)
                if first_angle is None:
                    first_angle = alpha

                next_obs_np = build_obs(s_next_np, goal_np, action_np)
                replay_buffer.add(obs_np, action_np, reward, next_obs_np, done)

                if len(replay_buffer) >= BATCH_SIZE and global_step >= LEARNING_STARTS and global_step % UPDATE_EVERY == 0:
                    batch = replay_buffer.sample(BATCH_SIZE, device)
                    latest_update = trainer.update(batch)

                pos_np = s_next_np
                vel_np = vel_new_np
                prev_action_np = action_np
                obs_np = next_obs_np
                global_step += 1
                steps += 1

                if done:
                    success = True
                    break

            ep_return = float(sum(rewards))
            dist_norm = (np.mean(dist_seq) / init_dist) if init_dist > 0 else 0.0
            mean_angle = float(np.mean(angle_seq)) if angle_seq else 0.0
            first_angle = first_angle if first_angle is not None else 0.0

            curve["ep"].append(global_ep)
            curve["phase"].append(phase_idx)
            curve["success"].append(float(success))
            curve["steps"].append(float(steps))
            curve["dist_norm"].append(float(dist_norm))
            curve["angle"].append(float(mean_angle))
            curve["first_angle"].append(float(first_angle))
            curve["return"].append(float(ep_return))
            curve["q_loss"].append(float(latest_update["q_loss"]))
            curve["actor_loss"].append(float(latest_update["actor_loss"]))
            curve["alpha"].append(float(latest_update["alpha"]))
            curve["q_target_mean"].append(float(latest_update["q_target_mean"]))
            curve["log_pi_mean"].append(float(latest_update["log_pi_mean"]))

            if local_ep % 50 == 0:
                w = min(50, len(curve["success"]))
                print(
                    f"  ep {local_ep:4d} | "
                    f"succ={np.mean(curve['success'][-w:]):.2f} | "
                    f"ret={np.mean(curve['return'][-w:]):.1f} | "
                    f"q_loss={np.mean(curve['q_loss'][-w:]):.4f} | "
                    f"actor={np.mean(curve['actor_loss'][-w:]):.4f} | "
                    f"alpha={np.mean(curve['alpha'][-w:]):.3f}",
                    flush=True,
                )

    return curve


def main():
    print(f"Device: {DEVICE}")
    os.makedirs(FIG_DIR, exist_ok=True)

    for seed in range(5):
        print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
        seed_dir = os.path.join(FIG_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        curve_path = os.path.join(seed_dir, "curve.npz")

        if os.path.exists(curve_path):
            print("Cached run found, loading.")
            curve = np.load(curve_path, allow_pickle=True)["curve"].item()
        else:
            curve = run_babab(seed=seed)
            np.savez(curve_path, curve=curve)

        plot_seed(seed, curve, GOAL_RADIUS, seed_dir)

    print(f"\nDone. Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
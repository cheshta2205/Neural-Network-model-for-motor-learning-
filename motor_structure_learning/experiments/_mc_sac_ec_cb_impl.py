"""Motor-learning SAC experiment with EC+CB augmented observations.

This variant extends the base SAC loop by adding:
- EC latent features from a recurrent transition encoder
- CB hidden features conditioned on state, previous action, and EC latent

Both EC and CB can optionally be initialized from a pretrained checkpoint.
"""

from __future__ import annotations

import argparse
import os
from collections import deque
import pdb
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from motor_structure_learning.core.ec import EntorhinalCortex
from motor_structure_learning.core.trainers import SACTrainer

from . import _mc_sac_impl as base

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    # elif torch.backends.mps.is_available():
    #     return "mps"
    else:
        return "cpu"


EC_HIDDEN = 256
EC_LATENT = 32
CB_HIDDEN = 256
CB_PROJ_DIM = 32

AUG_OBS_DIM = base.OBS_DIM + EC_LATENT + CB_PROJ_DIM

USE_PRETRAINED_EC_CB = True
FREEZE_PRETRAINED_EC_CB = True
EC_CB_CKPT = "checkpoints/ectocb_8ctx_seq80_z32.pt"

FIG_DIR = "figs/mc_sac_ec_cb"
DEVICE = get_device()

PRIORITY_GAMMA_T = 5000.0
PRIORITY_GAMMA_TD = 1.0
PRIORITY_MIX = 0.02


class CerebellumCB(nn.Module):
    """CB architecture compatible with historical EC->CB checkpoints."""

    def __init__(self, latent_dim: int = EC_LATENT, hidden: int = CB_HIDDEN):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(4 + latent_dim, hidden), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, 2)

    def hidden(self, state, action, z_ec):
        x = torch.cat([state, action, z_ec], dim=-1)
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        return h2

    def forward(self, state, action, z_ec):
        return self.out(self.hidden(state, action, z_ec))


def _freeze_module(module: nn.Module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def _remap_legacy_ec_state_dict(ec_state_dict):
    """Map legacy EC GRUCell keys (cell.*) to GRU keys (gru.*_l0)."""
    key_map = {
        "cell.weight_ih": "gru.weight_ih_l0",
        "cell.weight_hh": "gru.weight_hh_l0",
        "cell.bias_ih": "gru.bias_ih_l0",
        "cell.bias_hh": "gru.bias_hh_l0",
    }
    remapped = {}
    for key, value in ec_state_dict.items():
        remapped[key_map.get(key, key)] = value
    return remapped


def _remap_legacy_cb_state_dict(cb_state_dict):
    """Map legacy CB MLP keys (net.*) to CerebellumCB keys."""
    key_map = {
        "net.0.weight": "layer1.0.weight",
        "net.0.bias": "layer1.0.bias",
        "net.2.weight": "layer2.0.weight",
        "net.2.bias": "layer2.0.bias",
        "net.4.weight": "out.weight",
        "net.4.bias": "out.bias",
    }
    remapped = {}
    for key, value in cb_state_dict.items():
        remapped[key_map.get(key, key)] = value
    return remapped


def _load_ec_cb_if_available(ec: EntorhinalCortex, cb: CerebellumCB, ckpt_path: str, device: str) -> bool:
    if not os.path.exists(ckpt_path):
        print(f"EC/CB checkpoint not found at {ckpt_path}. Using random init.")
        return False

    ckpt = torch.load(ckpt_path, map_location=device)

    loaded_any = False
    if "ec" in ckpt:
        ec_sd = _remap_legacy_ec_state_dict(ckpt["ec"])
        ec.load_state_dict(ec_sd, strict=True)
        loaded_any = True
    else:
        print("Checkpoint has no 'ec' key. EC left randomly initialized.")

    if "cb" in ckpt:
        cb_sd = _remap_legacy_cb_state_dict(ckpt["cb"])
        cb.load_state_dict(cb_sd, strict=True)
        loaded_any = True
    else:
        print("Checkpoint has no 'cb' key. CB left randomly initialized.")

    if loaded_any:
        print(f"Loaded EC/CB weights from {ckpt_path}")
    return loaded_any


def _build_aug_obs(base_obs: np.ndarray, z_ec: torch.Tensor, z_cb: torch.Tensor) -> np.ndarray:
    z_ec_np = z_ec.detach().cpu().numpy().astype(np.float32).reshape(-1)
    z_cb_np = z_cb.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return np.concatenate([base_obs, z_ec_np, z_cb_np], axis=0).astype(np.float32)


def _to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)


class TimeTDEPrioritizedReplayBuffer:
    """Replay with p(e) proportional to exp(-(dt/gamma_t + |tde|/gamma_td))."""

    def __init__(self, capacity: int, gamma_t: float, gamma_td: float, mix: float = 0.0, seed: int = 0):
        self.capacity = capacity
        self.gamma_t = float(gamma_t)
        self.gamma_td = float(gamma_td)
        self.mix = float(mix)
        self.rng = np.random.default_rng(seed)

        self.observations = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_observations = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.timesteps = deque(maxlen=capacity)
        self.abs_tdes = deque(maxlen=capacity)

    def __len__(self):
        return len(self.observations)

    def add(self, obs, action, reward, next_obs, done, t_step: int, init_abs_tde: float = 0.0):
        self.observations.append(np.asarray(obs, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(np.asarray([reward], dtype=np.float32))
        self.next_observations.append(np.asarray(next_obs, dtype=np.float32).copy())
        self.dones.append(np.asarray([done], dtype=np.float32))
        self.timesteps.append(float(t_step))
        self.abs_tdes.append(float(init_abs_tde))

    def _sampling_probs(self, t_now: int) -> np.ndarray:
        dt = np.maximum(0.0, float(t_now) - np.asarray(self.timesteps, dtype=np.float64))
        abs_tde = np.asarray(self.abs_tdes, dtype=np.float64)

        energy = dt / max(self.gamma_t, 1e-8) + abs_tde / max(self.gamma_td, 1e-8)
        logits = -energy
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs_sum = probs.sum()
        if probs_sum <= 0.0 or not np.isfinite(probs_sum):
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        if self.mix > 0.0:
            probs = (1.0 - self.mix) * probs + self.mix * (1.0 / len(probs))
            probs = probs / probs.sum()

        return probs

    def sample(self, batch_size: int, device: str, t_now: int):
        probs = self._sampling_probs(t_now)
        idxs = self.rng.choice(len(self.observations), size=batch_size, replace=True, p=probs)

        observations = torch.tensor(np.stack([self.observations[i] for i in idxs]), device=device)
        actions = torch.tensor(np.stack([self.actions[i] for i in idxs]), device=device)
        rewards = torch.tensor(np.stack([self.rewards[i] for i in idxs]), device=device)
        next_observations = torch.tensor(np.stack([self.next_observations[i] for i in idxs]), device=device)
        dones = torch.tensor(np.stack([self.dones[i] for i in idxs]), device=device)

        batch = base.SACBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
        )
        return batch, idxs

    def update_abs_tde(self, idxs: np.ndarray, abs_tde_values: np.ndarray):
        for idx, val in zip(idxs, abs_tde_values):
            self.abs_tdes[int(idx)] = float(val)


def _current_features(
    ec: EntorhinalCortex,
    cb: CerebellumCB,
    cb_proj: torch.Tensor,
    h_ec: torch.Tensor,
    pos_np: np.ndarray,
    prev_action_np: np.ndarray,
    device: str,
):
    with torch.no_grad():
        z_ec = ec.readout(h_ec.squeeze(0))
        s_t = _to_tensor(pos_np, device)
        a_prev = _to_tensor(prev_action_np, device)
        cb_hidden = cb.hidden(s_t, a_prev, z_ec)
        z_cb = cb_hidden @ cb_proj
    return z_ec, z_cb


def _update_ec_hidden(
    ec: EntorhinalCortex,
    cb: CerebellumCB,
    h_ec: torch.Tensor,
    pos_np: np.ndarray,
    action_np: np.ndarray,
    next_pos_np: np.ndarray,
    device: str,
):
    with torch.no_grad():
        z_ec = ec.readout(h_ec.squeeze(0))
        s_t = _to_tensor(pos_np, device)
        a_t = _to_tensor(action_np, device)
        next_s_t = _to_tensor(next_pos_np, device)
        cb_pred_t = cb(s_t, a_t, z_ec)
        pred_err_t = cb_pred_t - next_s_t
        transition_t = torch.cat([s_t, a_t, next_s_t, pred_err_t], dim=-1).view(1, 1, -1)

        _, h_new = ec(transition_t, h_ec)
    return h_new

def run_babab(
    seed: int = 0,
    use_pretrained_ec_cb: bool = USE_PRETRAINED_EC_CB,
    freeze_pretrained_ec_cb: bool = FREEZE_PRETRAINED_EC_CB,
    ec_cb_ckpt: str = EC_CB_CKPT,
    priority_gamma_t: float = PRIORITY_GAMMA_T,
    priority_gamma_td: float = PRIORITY_GAMMA_TD,
    priority_mix: float = PRIORITY_MIX,
):
    device = DEVICE
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    ec = EntorhinalCortex(input_dim=8, hidden_dim=EC_HIDDEN, latent_dim=EC_LATENT).to(device)
    cb = CerebellumCB(latent_dim=EC_LATENT, hidden=CB_HIDDEN).to(device)

    if use_pretrained_ec_cb:
        _load_ec_cb_if_available(ec, cb, ec_cb_ckpt, device)

    if freeze_pretrained_ec_cb:
        _freeze_module(ec)
        _freeze_module(cb)

    cb_proj = torch.tensor(rng.standard_normal((CB_HIDDEN, CB_PROJ_DIM)), dtype=torch.float32, device=device)
    cb_proj = cb_proj / (cb_proj.norm(dim=0, keepdim=True) + 1e-8)

    actor = base.Actor(obs_dim=AUG_OBS_DIM, action_dim=base.ACTION_DIM).to(device)
    qf1 = base.SoftQNetwork(obs_dim=AUG_OBS_DIM, action_dim=base.ACTION_DIM).to(device)
    qf2 = base.SoftQNetwork(obs_dim=AUG_OBS_DIM, action_dim=base.ACTION_DIM).to(device)
    qf1_target = base.SoftQNetwork(obs_dim=AUG_OBS_DIM, action_dim=base.ACTION_DIM).to(device)
    qf2_target = base.SoftQNetwork(obs_dim=AUG_OBS_DIM, action_dim=base.ACTION_DIM).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=base.LR_ACTOR)
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=base.LR_Q)

    trainer = SACTrainer(
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        alpha=0.2,
        gamma=base.GAMMA,
        tau=base.TAU,
        target_entropy=-float(base.ACTION_DIM),
        autotune=True,
        device=device,
        grad_clip_norm=base.GRAD_CLIP_NORM,
    )

    replay_buffer = TimeTDEPrioritizedReplayBuffer(
        capacity=base.BUFFER_CAPACITY,
        gamma_t=priority_gamma_t,
        gamma_td=priority_gamma_td,
        mix=priority_mix,
        seed=seed,
    )

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
        "occurrence": [],
    }

    global_step = 0
    global_ep = 0

    for phase_idx, ctx in enumerate(base.PHASES):
        print(f"\nPhase {base.PHASE_LABELS[phase_idx]} ({base.N_PHASE_EP} episodes)...")

        for local_ep in range(1, base.N_PHASE_EP + 1):
            global_ep += 1

            beta, theta, gamma_curl, g_dyn, bx, by, _ = ctx
            pos_np = rng.uniform(0, base.GRID_SIZE, size=2).astype(np.float32)
            goal_np = rng.uniform(0, base.GRID_SIZE, size=2).astype(np.float32)
            while np.linalg.norm(pos_np - goal_np) < 0.5:
                goal_np = rng.uniform(0, base.GRID_SIZE, size=2).astype(np.float32)

            vel_np = np.zeros(2, dtype=np.float32)
            prev_action_np = np.zeros(2, dtype=np.float32)
            base_obs_np = base.build_obs(pos_np, goal_np, prev_action_np)
            init_dist = float(np.linalg.norm(pos_np - goal_np))

            h_ec = torch.zeros(1, 1, EC_HIDDEN, device=device)
            z_ec_t, z_cb_t = _current_features(ec, cb, cb_proj, h_ec, pos_np, prev_action_np, device)
            obs_np = _build_aug_obs(base_obs_np, z_ec_t, z_cb_t)

            rewards = []
            dist_seq = []
            angle_seq = []
            first_angle: Optional[float] = None
            success = False
            steps = 0
            latest_update = {
                "q_loss": 0.0,
                "actor_loss": 0.0,
                "alpha": trainer.current_alpha,
                "q_target_mean": 0.0,
                "log_pi_mean": 0.0,
            }

            for _ in range(base.MAX_STEPS):
                d = float(np.linalg.norm(pos_np - goal_np))
                dist_seq.append(d)
                if d < base.GOAL_RADIUS:
                    success = True
                    break

                obs_t = _to_tensor(obs_np, device)
                if global_step < base.LEARNING_STARTS or len(replay_buffer) < base.BATCH_SIZE:
                    action_np = rng.uniform(-1.0, 1.0, size=base.ACTION_DIM).astype(np.float32)
                else:
                    with torch.no_grad():
                        action_t, _, _ = trainer.act(obs_t)
                    action_np = action_t.squeeze(0).cpu().numpy().astype(np.float32)

                s_next_np, vel_new_np = base.env_step_np(
                    pos_np,
                    vel_np,
                    action_np,
                    beta,
                    theta,
                    gamma_curl,
                    g_dyn,
                    bx,
                    by,
                )
                done = bool(np.linalg.norm(s_next_np - goal_np) < base.GOAL_RADIUS)
                reward = -0.1 * float(np.linalg.norm(s_next_np - goal_np))
                if done:
                    reward += base.SUCCESS_BONUS
                rewards.append(reward)

                gd = goal_np - pos_np
                cross = gd[0] * action_np[1] - gd[1] * action_np[0]
                dot = action_np[0] * gd[0] + action_np[1] * gd[1]
                alpha = float(np.degrees(np.arctan2(cross, dot)))
                angle_seq.append(alpha)
                if first_angle is None:
                    first_angle = alpha

                h_ec = _update_ec_hidden(ec, cb, h_ec, pos_np, action_np, s_next_np, device)
                next_base_obs_np = base.build_obs(s_next_np, goal_np, action_np)
                next_z_ec_t, next_z_cb_t = _current_features(ec, cb, cb_proj, h_ec, s_next_np, action_np, device)
                next_obs_np = _build_aug_obs(next_base_obs_np, next_z_ec_t, next_z_cb_t)

                replay_buffer.add(obs_np, action_np, reward, next_obs_np, done, t_step=global_step)

                if (
                    len(replay_buffer) >= base.BATCH_SIZE
                    and global_step >= base.LEARNING_STARTS
                    and global_step % base.UPDATE_EVERY == 0
                ):
                    batch, idxs = replay_buffer.sample(base.BATCH_SIZE, device, t_now=global_step)

                    with torch.no_grad():
                        q_target = trainer.compute_q_target(batch)
                        q1 = trainer.qf1(batch.observations, batch.actions).view(-1)
                        q2 = trainer.qf2(batch.observations, batch.actions).view(-1)
                        abs_tde = 0.5 * (torch.abs(q1 - q_target) + torch.abs(q2 - q_target))

                    latest_update = trainer.update(batch)
                    replay_buffer.update_abs_tde(idxs, abs_tde.detach().cpu().numpy())

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
            curve["occurrence"].append(phase_idx // len(base.PHASE_DATA))

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


def main(
    seeds: int = 5,
    use_pretrained_ec_cb: bool = USE_PRETRAINED_EC_CB,
    freeze_pretrained_ec_cb: bool = FREEZE_PRETRAINED_EC_CB,
    ec_cb_ckpt: str = EC_CB_CKPT,
    fig_dir: str = FIG_DIR,
    priority_gamma_t: float = PRIORITY_GAMMA_T,
    priority_gamma_td: float = PRIORITY_GAMMA_TD,
    priority_mix: float = PRIORITY_MIX,
):
    print(f"Device: {DEVICE}")
    os.makedirs(fig_dir, exist_ok=True)

    print(
        "Config | "
        f"seeds={seeds}, "
        f"use_pretrained_ec_cb={use_pretrained_ec_cb}, "
        f"freeze_pretrained_ec_cb={freeze_pretrained_ec_cb}, "
        f"ec_cb_ckpt={ec_cb_ckpt}, "
        f"fig_dir={fig_dir}, "
        f"priority_gamma_t={priority_gamma_t}, "
        f"priority_gamma_td={priority_gamma_td}, "
        f"priority_mix={priority_mix}"
    )

    for seed in range(seeds):
        print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
        seed_dir = os.path.join(fig_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        curve_path = os.path.join(seed_dir, "curve.npz")

        if os.path.exists(curve_path):
            print("Cached run found, loading.")
            curve = np.load(curve_path, allow_pickle=True)["curve"].item()
        else:
            curve = run_babab(
                seed=seed,
                use_pretrained_ec_cb=use_pretrained_ec_cb,
                freeze_pretrained_ec_cb=freeze_pretrained_ec_cb,
                ec_cb_ckpt=ec_cb_ckpt,
                priority_gamma_t=priority_gamma_t,
                priority_gamma_td=priority_gamma_td,
                priority_mix=priority_mix,
            )
            np.savez(curve_path, curve=np.array(curve, dtype=object))

        key_params = {
            'use_pretrained_ec_cb': use_pretrained_ec_cb,
            'freeze_pretrained_ec_cb': freeze_pretrained_ec_cb,
            'priority_gamma_t': priority_gamma_t,
            'priority_gamma_td': priority_gamma_td,
            'priority_mix': priority_mix,
        }
        base.plot_seed(seed, curve, base.GOAL_RADIUS, seed_dir, key_params=key_params)

    print(f"\nDone. Figures saved to {fig_dir}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train SAC with EC+CB-augmented observations.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to run.")
    parser.add_argument(
        "--ec-cb-ckpt",
        type=str,
        default=EC_CB_CKPT,
        help="Path to checkpoint with optional 'ec' and 'cb' state dicts.",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=FIG_DIR,
        help="Output directory for figures and cached curves.",
    )
    parser.add_argument(
        "--priority-gamma-t",
        type=float,
        default=PRIORITY_GAMMA_T,
        help="Recency temperature in exp(-(dt/gamma_t + |tde|/gamma_td)).",
    )
    parser.add_argument(
        "--priority-gamma-td",
        type=float,
        default=PRIORITY_GAMMA_TD,
        help="TD-error temperature in exp(-(dt/gamma_t + |tde|/gamma_td)).",
    )
    parser.add_argument(
        "--priority-mix",
        type=float,
        default=PRIORITY_MIX,
        help="Uniform-mixture fraction for replay probabilities.",
    )

    parser.set_defaults(use_pretrained_ec_cb=USE_PRETRAINED_EC_CB)
    parser.add_argument(
        "--use-pretrained-ec-cb",
        dest="use_pretrained_ec_cb",
        action="store_true",
        help="Enable loading EC/CB from checkpoint.",
    )
    parser.add_argument(
        "--no-use-pretrained-ec-cb",
        dest="use_pretrained_ec_cb",
        action="store_false",
        help="Disable loading EC/CB from checkpoint.",
    )

    parser.set_defaults(freeze_pretrained_ec_cb=FREEZE_PRETRAINED_EC_CB)
    parser.add_argument(
        "--freeze-pretrained-ec-cb",
        dest="freeze_pretrained_ec_cb",
        action="store_true",
        help="Freeze EC/CB parameters during SAC training.",
    )
    parser.add_argument(
        "--no-freeze-pretrained-ec-cb",
        dest="freeze_pretrained_ec_cb",
        action="store_false",
        help="Allow EC/CB parameters to remain trainable (not optimized by SAC by default).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        seeds=args.seeds,
        use_pretrained_ec_cb=args.use_pretrained_ec_cb,
        freeze_pretrained_ec_cb=args.freeze_pretrained_ec_cb,
        ec_cb_ckpt=args.ec_cb_ckpt,
        fig_dir=args.fig_dir,
        priority_gamma_t=args.priority_gamma_t,
        priority_gamma_td=args.priority_gamma_td,
        priority_mix=args.priority_mix,
    )

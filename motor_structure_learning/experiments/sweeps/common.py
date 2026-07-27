"""Shared building blocks for every ablation-sweep variant (feedforward/
recurrent actor x base/instructed-delay task).

Everything here is architecture- and task-agnostic: EC/CB augmented
observations, the prioritized replay buffer, checkpoint (de)serialization, and
the shared context/config definitions. The variant-specific runner modules
(`feedforward_base`, `feedforward_delay`, `recurrent_base`, `recurrent_delay`)
each import this module as `sweep` and add only what actually differs for
that combination (actor architecture, episode/rollout mechanics).
"""

from __future__ import annotations

import os
import re
from collections import deque
import pdb
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from motor_structure_learning.core.ec import EntorhinalCortex
from motor_structure_learning.core.trainers import SACTrainer, ECCBTrainer

from motor_structure_learning.experiments import _mc_sac_impl as base


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    # elif torch.backends.mps.is_available():
    #     return "mps"
    else:
        return "cpu"


CTX_A = (1.0, np.pi / 3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B = (1.0, 0.0, +0.4, 0.0, 0.0, 0.0, "B: curl+")
CTX_C = (1.0, -np.pi / 3, 0.0, 0.0, 0.0, 0.0, "C: rot-60")
CTX_D = (1.0, 0.0, -0.4, 0.0, 0.0, 0.0, "D: curl-")


PHASE_REPS = 3

PHASE_DATA = [
    {"label": "A (rot +60°)", "params": CTX_A},
    {"label": "B (curl+)", "params": CTX_B},
    {"label": "C (rot -60°)", "params": CTX_C},
    {"label": "D (curl-)", "params": CTX_D}
]

N_PHASES = len(PHASE_DATA) * PHASE_REPS

base.N_PHASE_EP = 500
base.PHASE_DATA = PHASE_DATA
base.N_PHASES = N_PHASES
base.PHASE_REPS = PHASE_REPS

EC_HIDDEN = 256
EC_LATENT = 32
CB_HIDDEN = 256
CB_PROJ_DIM = 32
ORACLE_CTX_DIM = 32

USE_PRETRAINED_EC_CB = True
FREEZE_PRETRAINED_EC_CB = False
EC_CB_CKPT = "checkpoints/ectocb_8ctx_seq80_z32.pt"

DEVICE = get_device()

PRIORITY_GAMMA_T = 5000.0
PRIORITY_GAMMA_TD = 1.0
PRIORITY_MIX = 0.02

# Define Configurations

CONFIG_PRESETS = [
    {"name": "baseline", "desc": "Motor cortex trained with SAC no EC/CB",
     "use_ec": False, "use_cb": False, "use_pretrained_ec_cb": False, "freeze_pretrained_ec_cb": False,
     "priority_gamma_t": 1000.0, "priority_gamma_td": 1.0, "priority_mix": 0.02, "use_oracle": False},
    {"name": "full_end_to_end", "desc": "Add both EC and CB features, with pretrained weights, no freezing, no recency bias",
     "use_ec": True, "use_cb": True, "use_pretrained_ec_cb": False, "freeze_pretrained_ec_cb": False,
     "priority_gamma_t": 1000.0, "priority_gamma_td": 1.0, "priority_mix": 0.02, "use_oracle": False},
    {"name": "oracle", "desc": "Add one-hot context encoding of the current phase, no EC/CB",
     "use_ec": False, "use_cb": False, "use_pretrained_ec_cb": False, "freeze_pretrained_ec_cb": False,
     "priority_gamma_t": 1000.0, "priority_gamma_td": 1.0, "priority_mix": 0.02, "use_oracle": True},
     {"name": "full", "desc": "Add both EC and CB features, with frozen pretrained weights",
     "use_ec": True, "use_cb": True, "use_pretrained_ec_cb": True, "freeze_pretrained_ec_cb": True,
     "priority_gamma_t": 1000.0, "priority_gamma_td": 1.0, "priority_mix": 0.02, "use_oracle": False},
    {"name": "cb_only", "desc": "Add CB features, no EC, no recency bias",
     "use_ec": False, "use_cb": True, "use_pretrained_ec_cb": True, "freeze_pretrained_ec_cb": False,
     "priority_gamma_t": 1000.0, "priority_gamma_td": 1.0, "priority_mix": 0.02, "use_oracle": False},
]


def _compute_aug_obs_dim(use_ec: bool, use_cb: bool, use_oracle_ctx: bool) -> int:
    """Compute observation dimension based on which modules are enabled."""
    dim = base.OBS_DIM
    if use_ec:
        dim += EC_LATENT
    if use_cb:
        dim += CB_PROJ_DIM
    if use_oracle_ctx:
        dim += ORACLE_CTX_DIM
    return dim


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
    if "ec" in ckpt and ec is not None:
        ec_sd = _remap_legacy_ec_state_dict(ckpt["ec"])
        ec.load_state_dict(ec_sd, strict=True)
        loaded_any = True
    else:
        print("Checkpoint has no 'ec' key. EC left randomly initialized.")

    if "cb" in ckpt and cb is not None:
        cb_sd = _remap_legacy_cb_state_dict(ckpt["cb"])
        cb.load_state_dict(cb_sd, strict=True)
        loaded_any = True
    else:
        print("Checkpoint has no 'cb' key. CB left randomly initialized.")

    if loaded_any:
        print(f"Loaded EC/CB weights from {ckpt_path}")
    return loaded_any

def _build_aug_obs(
    base_obs: np.ndarray,
    z_ec: Optional[torch.Tensor],
    z_cb: Optional[torch.Tensor],
    use_ec: bool,
    use_cb: bool,
    use_oracle: bool,
    ctx_one_hot: Optional[np.ndarray] = None
) -> np.ndarray:
    """Build augmented observation, conditionally including EC and CB features."""
    components = [base_obs]

    if use_ec and z_ec is not None:
        z_ec_np = z_ec.detach().cpu().numpy().astype(np.float32).reshape(-1)
        components.append(z_ec_np)

    if use_cb and z_cb is not None:
        z_cb_np = z_cb.detach().cpu().numpy().astype(np.float32).reshape(-1)
        components.append(z_cb_np)

    if use_oracle and ctx_one_hot is not None:
        components.append(ctx_one_hot.astype(np.float32).reshape(-1))

    return np.concatenate(components, axis=0).astype(np.float32)


def _to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)


class TimeTDEPrioritizedReplayBuffer:
    """Replay with p(e) proportional to exp(-(dt/gamma_t + |tde|/gamma_td))."""

    def __init__(self, capacity: int, gamma_t: float, gamma_td: float, mix: float = 0.0, seed: int = 0, ec_hidden_dim: int = 256):
        self.capacity = capacity
        self.gamma_t = float(gamma_t)
        self.gamma_td = float(gamma_td)
        self.mix = float(mix)
        self.rng = np.random.default_rng(seed)
        self.ec_hidden_dim = ec_hidden_dim

        self.observations = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_observations = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.timesteps = deque(maxlen=capacity)
        self.abs_tdes = deque(maxlen=capacity)

        # Transition data for EC replay
        self.s_t = deque(maxlen=capacity)           # current position
        self.a_t = deque(maxlen=capacity)           # current action
        self.s_t_next = deque(maxlen=capacity)      # next position
        self.pred_err_t = deque(maxlen=capacity)    # CB prediction error
        self.h_ec_prev = deque(maxlen=capacity)     # previous EC hidden state

    def __len__(self):
        return len(self.observations)

    def add(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        t_step: int,
        init_abs_tde: float = 0.0,
        s_t: Optional[np.ndarray] = None,
        s_t_next: Optional[np.ndarray] = None,
        pred_err_t: Optional[np.ndarray] = None,
        h_ec: Optional[torch.Tensor] = None,
    ):
        self.observations.append(np.asarray(obs, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(np.asarray([reward], dtype=np.float32))
        self.next_observations.append(np.asarray(next_obs, dtype=np.float32).copy())
        self.dones.append(np.asarray([done], dtype=np.float32))
        self.timesteps.append(float(t_step))
        self.abs_tdes.append(float(init_abs_tde))

        # Store transition data for EC replay
        self.s_t.append(np.asarray(s_t if s_t is not None else np.zeros(2), dtype=np.float32).copy())
        self.s_t_next.append(np.asarray(s_t_next if s_t_next is not None else np.zeros(2), dtype=np.float32).copy())
        self.pred_err_t.append(np.asarray(pred_err_t if pred_err_t is not None else np.zeros(2), dtype=np.float32).copy())

        # Store EC hidden state for exact replay
        if h_ec is not None:
            self.h_ec_prev.append(h_ec.detach().cpu().numpy().astype(np.float32).copy())
        else:
            self.h_ec_prev.append(np.zeros((1, self.ec_hidden_dim), dtype=np.float32))

    def _sampling_probs(self, t_now: int) -> np.ndarray:
        dt = np.maximum(0.0, float(t_now) - np.asarray(self.timesteps, dtype=np.float64))

        energy = dt / max(self.gamma_t, 1e-8)
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

        # Transition data for EC replay
        s_t = torch.tensor(np.stack([self.s_t[i] for i in idxs]), device=device)
        a_t = torch.tensor(np.stack([self.actions[i] for i in idxs]), device=device)
        s_t_next = torch.tensor(np.stack([self.s_t_next[i] for i in idxs]), device=device)
        pred_err_t = torch.tensor(np.stack([self.pred_err_t[i] for i in idxs]), device=device)
        h_ec_prev = torch.tensor(np.stack([self.h_ec_prev[i] for i in idxs]), device=device)

        h_ec_prev = h_ec_prev.permute(1, 0, 2).contiguous()

        batch = base.SACBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
        )
        return batch, idxs, (s_t, a_t, s_t_next, pred_err_t, h_ec_prev)

    def update_abs_tde(self, idxs: np.ndarray, abs_tde_values: np.ndarray):
        for idx, val in zip(idxs, abs_tde_values):
            self.abs_tdes[int(idx)] = float(val)


def _current_features(
    ec: Optional[EntorhinalCortex],
    cb: Optional[CerebellumCB],
    cb_proj: Optional[torch.Tensor],
    h_ec: torch.Tensor,
    pos_np: np.ndarray,
    prev_action_np: np.ndarray,
    device: str,
    use_ec: bool,
    use_cb: bool,
):
    """Extract EC and CB features, returning None for disabled modules."""
    z_ec = None
    z_cb = None

    with torch.no_grad():
        if use_ec and ec is not None:
            z_ec = ec.readout(h_ec.squeeze(0))
        else:
            z_ec = torch.zeros(1, EC_LATENT, device=device)

        if use_cb and cb is not None and cb_proj is not None:
            s_t = _to_tensor(pos_np, device)
            a_prev = _to_tensor(prev_action_np, device)
            cb_hidden = cb.hidden(s_t, a_prev, z_ec)
            z_cb = cb_hidden @ cb_proj
        else:
            z_cb = torch.zeros(1, CB_PROJ_DIM, device=device)

    return z_ec, z_cb


def _update_ec_hidden(
    ec: Optional[EntorhinalCortex],
    cb: Optional[CerebellumCB],
    h_ec: torch.Tensor,
    pos_np: np.ndarray,
    action_np: np.ndarray,
    next_pos_np: np.ndarray,
    device: str,
    use_ec: bool,
):
    """Update EC hidden state. Skip if use_ec=False."""
    if not use_ec or ec is None:
        return h_ec, torch.zeros(1, 2, device=device)

    with torch.no_grad():
        z_ec = ec.readout(h_ec.squeeze(0))
        s_t = _to_tensor(pos_np, device)
        a_t = _to_tensor(action_np, device)
        next_s_t = _to_tensor(next_pos_np, device)
        cb_pred_t = cb(s_t, a_t, z_ec) if cb is not None else torch.zeros(1, 2, device=device)
        pred_err_t = cb_pred_t - next_s_t
        transition_t = torch.cat([s_t, a_t, next_s_t, pred_err_t], dim=-1).view(1, 1, -1)

        _, h_new = ec(transition_t, h_ec)
    return h_new, pred_err_t

def _context_one_hot(phase_id: int) -> np.ndarray:
    """Return one-hot encoding of the current phase_id for oracle ablation."""
    v = np.zeros(ORACLE_CTX_DIM, dtype=np.float32)
    v[int(phase_id)] = 1
    return v


def _slugify(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def _save_phase_checkpoint(
    checkpoint_dir: str,
    phase_idx: int,
    phase_label: str,
    trainer: SACTrainer,
    ec_cb_trainer: Optional[ECCBTrainer],
    seed: int,
    config_name: Optional[str],
    global_step: int,
    global_ep: int,
) -> str:
    """Save actor/critics/EC/CB weights (+ optimizers) after a training block."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = {
        "phase_idx": phase_idx,
        "phase_label": phase_label,
        "seed": seed,
        "config_name": config_name,
        "global_step": global_step,
        "global_ep": global_ep,
        "trainer": trainer.state_dict(),
        "ec_cb_trainer": ec_cb_trainer.state_dict() if ec_cb_trainer is not None else None,
    }
    path = os.path.join(checkpoint_dir, f"phase_{phase_idx:02d}_{_slugify(phase_label)}.pt")
    torch.save(ckpt, path)
    return path


def load_phase_checkpoint(
    ckpt_path: str,
    trainer: SACTrainer,
    ec_cb_trainer: Optional[ECCBTrainer] = None,
    map_location: Optional[str] = None,
) -> dict:
    """Load a saved phase checkpoint's weights into already-constructed trainer(s).

    Returns the raw checkpoint dict so callers can read metadata (phase_idx,
    phase_label, seed, config_name, global_step, global_ep).
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    trainer.load_state_dict(ckpt["trainer"])
    if ec_cb_trainer is not None and ckpt.get("ec_cb_trainer") is not None:
        ec_cb_trainer.load_state_dict(ckpt["ec_cb_trainer"])
    return ckpt

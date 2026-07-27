"""EC/CB prediction trainer for motor learning experiments.

This trainer handles the supervised learning objective of training the
entorhinal cortex (EC) and cerebellum (CB) models to predict the next
state from the current state, action, and EC latent representation.

Uses sequence replay: stores transition components and replays through EC
to compute h_ec with full gradient flow for training EC RNN weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import pdb
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from .base import BaseTrainer


@dataclass(slots=True)
class ECCBBatch:
    """Batch for EC/CB prediction training."""

    observations: torch.Tensor      # [B, obs_dim] (includes z_ec, z_cb)
    actions: torch.Tensor           # [B, action_dim]
    next_observations: torch.Tensor # [B, obs_dim]
    
    # Transition data for EC replay
    s_t: Optional[torch.Tensor] = None           # [B, 2] current position
    a_t: Optional[torch.Tensor] = None           # [B, 2] current action
    s_t_next: Optional[torch.Tensor] = None      # [B, 2] next position
    pred_err_t: Optional[torch.Tensor] = None    # [B, 2] CB prediction error
    h_ec_prev: Optional[torch.Tensor] = None     # [B, 1, ec_hidden_dim] previous EC hidden state


class ECCBTrainer(BaseTrainer):
    """Trainer for EC/CB backbone prediction using sequence replay."""

    def __init__(
        self,
        ec: Optional[nn.Module] = None,
        cb: Optional[nn.Module] = None,
        cb_proj: Optional[torch.Tensor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        base_obs_dim: int = 6,
        ec_latent_dim: int = 32,
        cb_proj_dim: int = 32,
        ec_hidden_dim: int = 256,
        use_ec: bool = True,
        use_cb: bool = True,
        device: Optional[torch.device | str] = None,
        grad_clip_norm: Optional[float] = None,
    ) -> None:
        """Initialize EC/CB trainer.
        
        Args:
            ec: EntorhinalCortex model (can be None if use_ec=False)
            cb: CerebellumCB model (can be None if use_cb=False)
            cb_proj: Projection matrix for CB outputs (can be None if use_cb=False)
            optimizer: Optimizer for trainable parameters
            base_obs_dim: Dimension of base observation (pos + goal + prev_action)
            ec_latent_dim: Dimension of EC latent output
            cb_proj_dim: Dimension of CB projection
            ec_hidden_dim: Dimension of EC hidden state
            use_ec: Whether EC is active
            use_cb: Whether CB is active
            device: Device to place models on
            grad_clip_norm: Gradient clipping threshold
        """
        models = {}
        if ec is not None:
            models["ec"] = ec
        if cb is not None:
            models["cb"] = cb

        optimizers = {}
        if optimizer is not None:
            optimizers["ec_cb"] = optimizer

        super().__init__(models=models, optimizers=optimizers, device=device, grad_clip_norm=grad_clip_norm)

        self.ec = ec
        self.cb = cb
        self.cb_proj = cb_proj
        self.optimizer = optimizer

        self.base_obs_dim = base_obs_dim
        self.ec_latent_dim = ec_latent_dim
        self.cb_proj_dim = cb_proj_dim
        self.ec_hidden_dim = ec_hidden_dim
        self.use_ec = use_ec
        self.use_cb = use_cb

    def _extract_state_and_features(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract base state and optional EC/CB features from observation.
        
        Observation layout: [base_obs (6) | z_ec (32) | z_cb (32)]
        Returns: (base_state, z_ec, z_cb) where base_state is first 2 dims (pos)
        """
        base_obs = observations[:, :self.base_obs_dim]
        base_state = base_obs[:, :2]  # pos is first 2 entries
        
        z_ec = None
        z_cb = None
        
        ptr = self.base_obs_dim
        if self.use_ec:
            z_ec = observations[:, ptr : ptr + self.ec_latent_dim]
            ptr += self.ec_latent_dim
        if self.use_cb:
            z_cb = observations[:, ptr : ptr + self.cb_proj_dim]
            ptr += self.cb_proj_dim
        
        return base_state, z_ec, z_cb

    def replay_ec_sequence(self, batch: ECCBBatch) -> torch.Tensor:
        if self.ec is None or batch.s_t is None:
            bsz = batch.observations.shape[0]
            return torch.zeros(bsz, self.ec_latent_dim, device=batch.observations.device)

        # Initial hidden
        if batch.h_ec_prev is not None:
            h_ec_init = batch.h_ec_prev
            if h_ec_init.dim() == 3 and h_ec_init.shape[0] != 1:
                # if [B,1,H] accidentally comes in, convert to [1,B,H]
                h_ec_init = h_ec_init.permute(1, 0, 2).contiguous()
        else:
            bsz = batch.s_t.shape[0]
            h_ec_init = torch.zeros(1, bsz, self.ec_hidden_dim, device=batch.s_t.device, dtype=batch.s_t.dtype)

        # Recompute pred_err_t with grad
        # z_prev from replayed hidden context (gradient flows into EC readout params)
        z_prev = self.ec.readout(h_ec_init.squeeze(0))  # [B,H] -> [B,latent]

        if self.use_cb and self.cb is not None:
            cb_pred_prev = self.cb(batch.s_t, batch.a_t, z_prev)   # [B,2]
            pred_err_t = cb_pred_prev - batch.s_t_next             # [B,2]
        else:
            # fallback if CB disabled
            pred_err_t = batch.pred_err_t if batch.pred_err_t is not None else torch.zeros_like(batch.s_t_next)

        transition = torch.cat([batch.s_t, batch.a_t, batch.s_t_next, pred_err_t], dim=-1).unsqueeze(1)  # [B,1,8]

        _, h_new = self.ec(transition, h_ec_init)
        z_ec = self.ec.readout(h_new.squeeze(0))
        return z_ec

    def compute_loss(self, batch: ECCBBatch) -> dict[str, torch.Tensor]:
        losses = {}

        if batch.s_t is None or batch.a_t is None or batch.s_t_next is None:
            total_loss = torch.tensor(0.0, device=batch.observations.device)
            return {"total_loss": total_loss}

        # If EC is enabled, replay to get z_ec with gradient.
        if self.use_ec:
            z_ec = self.replay_ec_sequence(batch)
        else:
            # cb_only path: provide a deterministic zero EC context so CB can still train.
            z_ec = torch.zeros(
                batch.s_t.shape[0],
                self.ec_latent_dim,
                device=batch.s_t.device,
                dtype=batch.s_t.dtype,
            )

        if self.cb is not None:
            cb_pred = self.cb(batch.s_t, batch.a_t, z_ec)
            cb_loss = F.mse_loss(cb_pred, batch.s_t_next)
            losses["cb_loss"] = cb_loss

        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=batch.observations.device)
        losses["total_loss"] = total_loss
        return losses

    def update(self, batch: ECCBBatch) -> dict[str, float]:
        """Run one EC/CB training step with sequence replay.
        
        Args:
            batch: Batch with observations and transition data (s_t, a_t, s_t_next, pred_err_t)
            
        Returns:
            Dict of scalar diagnostics
        """
        if self.optimizer is None:
            return {"ec_cb_loss": np.nan, "cb_loss": np.nan}

        losses = self.compute_loss(batch)
        total_loss = losses["total_loss"]

        # Backward pass (now gradients flow through EC!)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.clip_grad_norm_()
        self.optimizer.step()

        result = {
            "ec_cb_loss": float(total_loss.detach().item()),
            "cb_loss": float(losses.get("cb_loss", torch.nan).detach().item()),
        }

        return result

    def state_dict(self) -> dict[str, Any]:
        """Extend the base checkpoint with the (non-module) CB projection matrix."""
        state = super().state_dict()
        if self.cb_proj is not None:
            state["cb_proj"] = self.cb_proj.detach().cpu()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore models/optimizers plus `cb_proj` from a checkpoint."""
        super().load_state_dict(state)
        if self.cb_proj is not None and "cb_proj" in state:
            self.cb_proj.copy_(state["cb_proj"].to(self.cb_proj.device))
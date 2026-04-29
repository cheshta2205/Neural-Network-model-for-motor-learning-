"""EC/CB prediction trainer for motor learning experiments.

This trainer handles the supervised learning objective of training the
entorhinal cortex (EC) and cerebellum (CB) models to predict the next
state from the current state, action, and EC latent representation.

Uses sequence replay: stores transition components and replays through EC
to compute h_ec with full gradient flow for training EC RNN weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import pdb
from typing import Optional

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
        """Replay transitions through EC using stored h_ec context.
        
        This is an *exact* replay: uses the previous h_ec state and processes
        the transition through the RNN, allowing gradients to flow through
        the RNN weights.
        
        Args:
            batch: Must have s_t, a_t, s_t_next, pred_err_t, and h_ec_prev populated
            
        Returns:
            z_ec: [B, ec_latent_dim] latent representations from EC readout
        """
        if self.ec is None or batch.s_t is None:
            batch_size = batch.s_t.shape[0] if batch.s_t is not None else batch.observations.shape[0]
            return torch.zeros(batch_size, self.ec_latent_dim, device=batch.observations.device)
        
        # Build transition tensor: [s_t, a_t, s_t+1, pred_err_t]
        # Shape: [B, 1, 8] (for EC input_dim=8)
        transition = torch.cat([batch.s_t, batch.a_t, batch.s_t_next, batch.pred_err_t], dim=-1)
        transition = transition.unsqueeze(1)  # [B, 1, 8]
        
        # Use stored h_ec_prev as initial state for exact replay
        if batch.h_ec_prev is not None:
            h_ec_init = batch.h_ec_prev
        else:
            # Fallback to zero if not provided
            batch_size = transition.shape[0]
            h_ec_init = torch.zeros(1, batch_size, self.ec_hidden_dim, device=transition.device, dtype=transition.dtype)
        
        # Forward through EC RNN with proper initial state
        _, h_new = self.ec(transition, h_ec_init)  # h_new: [B, 1, hidden]

        # Extract latent via readout
        z_ec = self.ec.readout(h_new.squeeze(0))  # [B, hidden] -> [B, latent]
        
        return z_ec

    def compute_loss(self, batch: ECCBBatch) -> dict[str, torch.Tensor]:
        """Compute EC/CB prediction loss with EC sequence replay.
        
        Returns:
            Dict with 'total_loss' and 'cb_loss'
        """
        losses = {}
        
        # Extract target position
        next_s_t = batch.s_t_next if batch.s_t_next is not None else torch.zeros_like(batch.s_t)
        
        # Replay EC to get z_ec (with gradients!)
        if self.use_ec and batch.s_t is not None:
            z_ec = self.replay_ec_sequence(batch)
        else:
            z_ec = self._extract_state_and_features(batch.observations)[1]  # fallback to stored
        
        # CB prediction loss: CB(s_t, a_t, z_ec) -> s_{t+1}
        if self.cb is not None and z_ec is not None and batch.s_t is not None:
            cb_pred = self.cb(batch.s_t, batch.a_t, z_ec)  # [B, 2]
            cb_loss = F.mse_loss(cb_pred, next_s_t)
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
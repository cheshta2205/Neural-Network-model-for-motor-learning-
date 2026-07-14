"""Soft Actor-Critic trainer skeleton.

This follows the same broad structure as CleanRL's single-file SAC implementation:

- actor and twin Q networks are owned by the trainer
- replay buffer batches are passed into `update`
- target networks are soft-updated inside the trainer
- rollout collection remains outside the trainer

The class is intentionally thin so experiment scripts can keep the CleanRL-style
top-level flow while reusing the update mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseTrainer


@dataclass(slots=True)
class SACBatch:
    """Transition batch expected by the SAC trainer."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


SACActor = Any
SACQNetwork = Any

class SACTrainer(BaseTrainer):
    """CleanRL-style SAC trainer for continuous-action experiments."""

    def __init__(
        self,
        actor: nn.Module,
        qf1: nn.Module,
        qf2: nn.Module,
        qf1_target: nn.Module,
        qf2_target: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_optimizer: torch.optim.Optimizer,
        alpha_optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        alpha: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        autotune: bool = True,
        device: Optional[torch.device | str] = None,
        grad_clip_norm: Optional[float] = None,
    ) -> None:
        resolved_device = device
        if resolved_device is None:
            try:
                resolved_device = next(actor.parameters()).device
            except StopIteration:
                resolved_device = torch.device("cpu")

        models = {
            "actor": actor,
            "qf1": qf1,
            "qf2": qf2,
            "qf1_target": qf1_target,
            "qf2_target": qf2_target,
        }

        optimizers = {
            "actor": actor_optimizer,
            "q": q_optimizer,
        }
        
        if alpha_optimizer is not None:
            optimizers["alpha"] = alpha_optimizer

        super().__init__(models=models, optimizers=optimizers, device=resolved_device, grad_clip_norm=grad_clip_norm)

        self.actor = actor
        self.qf1 = qf1
        self.qf2 = qf2
        self.qf1_target = qf1_target
        self.qf2_target = qf2_target
        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer
        self.alpha_optimizer = alpha_optimizer

        self.gamma = gamma
        self.tau = tau
        self.autotune = autotune
        self.target_entropy = target_entropy

        self.log_alpha: Optional[torch.Tensor]
        if autotune:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            if self.alpha_optimizer is None:
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
                self.register_optimizer("alpha", self.alpha_optimizer)
            else:
                self.register_optimizer("alpha", self.alpha_optimizer)
        else:
            self.log_alpha = None

        self.alpha = float(alpha)

    @property
    def current_alpha(self) -> float:
        """Return the scalar entropy temperature used for actor updates."""
        if self.autotune and self.log_alpha is not None:
            return float(self.log_alpha.exp().item())
        return self.alpha

    def act(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action the CleanRL way: actor returns action, log_prob, mean."""
        get_action = getattr(self.actor, "get_action")
        return get_action(observations)

    def compute_q_target(self, batch: SACBatch) -> torch.Tensor:
        """Build the soft Bellman target using target Q networks."""
        with torch.no_grad():
            get_action = getattr(self.actor, "get_action")
            next_actions, next_log_pi, _ = get_action(batch.next_observations)
            qf1_next_target = self.qf1_target(batch.next_observations, next_actions)
            qf2_next_target = self.qf2_target(batch.next_observations, next_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.current_alpha * next_log_pi
            return batch.rewards.flatten() + (1.0 - batch.dones.flatten()) * self.gamma * min_qf_next_target.view(-1)

    def compute_q_losses(self, batch: SACBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return qf1 loss, qf2 loss, and the shared target value."""
        next_q_value = self.compute_q_target(batch)
        qf1_a_values = self.qf1(batch.observations, batch.actions).view(-1)
        qf2_a_values = self.qf2(batch.observations, batch.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        return qf1_loss, qf2_loss, next_q_value

    def compute_actor_loss(self, batch: SACBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the actor loss and the sampled log-probability."""
        get_action = getattr(self.actor, "get_action")
        pi, log_pi, _ = get_action(batch.observations)
        qf1_pi = self.qf1(batch.observations, pi)
        qf2_pi = self.qf2(batch.observations, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.current_alpha * log_pi) - min_qf_pi).mean()
        return actor_loss, log_pi

    def compute_alpha_loss(self, log_pi: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the temperature loss when automatic tuning is enabled."""
        if not self.autotune or self.log_alpha is None or self.target_entropy is None:
            return None
        return (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

    def soft_update_targets(self) -> None:
        """Polyak-average Q targets from the online critics."""
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update(self, batch: SACBatch) -> dict[str, float]:
        """Run one SAC update step and return scalar diagnostics.

        This keeps the same division of labor as CleanRL:
        critic update -> actor update -> optional alpha update -> target update.
        """
        qf1_loss, qf2_loss, q_target = self.compute_q_losses(batch)
        q_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        actor_loss, log_pi = self.compute_actor_loss(batch)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = self.compute_alpha_loss(log_pi)
        if alpha_loss is not None and self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.soft_update_targets()

        return {
            "qf1_loss": float(qf1_loss.detach().item()),
            "qf2_loss": float(qf2_loss.detach().item()),
            "q_loss": float(q_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "alpha": float(self.current_alpha),
            "alpha_loss": float(alpha_loss.detach().item()) if alpha_loss is not None else 0.0,
            "q_target_mean": float(q_target.detach().mean().item()),
            "log_pi_mean": float(log_pi.detach().mean().item()),
        }

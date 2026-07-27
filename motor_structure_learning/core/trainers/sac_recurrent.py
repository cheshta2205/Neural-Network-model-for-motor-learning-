"""SAC trainer variant for a recurrent actor.

Off-policy replay makes recurrent actors awkward: `SACTrainer.compute_q_target`
and `compute_actor_loss` recompute actions from a *randomly sampled* batch, so
a stateless actor just needs the observation -- but a recurrent one also
needs the hidden state that was actually present at that point in the
trajectory. This mirrors the same problem `ECCBTrainer` already solves for
`EntorhinalCortex`'s hidden state: the replay buffer stores the hidden state
entering each transition (`h_actor_prev`), and it's replayed one step forward
from there rather than assumed to be zero or recomputed from scratch.

Everything else (critic losses, target soft-updates, checkpoint format) is
inherited unchanged from `SACTrainer` -- only the three methods that call the
actor are overridden.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sac import SACBatch, SACTrainer


@dataclass(slots=True)
class RecurrentSACBatch(SACBatch):
    """`SACBatch` plus the recurrent actor's hidden state entering
    `observations` (shape `(batch, hidden_dim)`), needed to replay the actor
    correctly rather than assuming a stateless reaction to the observation.
    """

    h_actor_prev: torch.Tensor


class RecurrentSACTrainer(SACTrainer):
    """`SACTrainer` for a `RecurrentActor` (see `_mc_sac_impl.RecurrentActor`).

    `actor.get_action(obs, h)` returns `(action, log_prob, mean, h_new)` --
    one extra element and one extra argument versus the stateless actor's
    `get_action(obs)` -- so `act`/`compute_q_target`/`compute_actor_loss` are
    overridden to thread `h_actor_prev` through. `compute_q_losses`, `update`,
    `soft_update_targets`, and checkpoint (de)serialization are all inherited
    unchanged from `SACTrainer`.
    """

    def act(self, observations: torch.Tensor, h_actor: torch.Tensor):
        """Sample an action: actor returns `(action, log_prob, mean, h_new)`."""
        return self.actor.get_action(observations, h_actor)

    def compute_q_target(self, batch: RecurrentSACBatch) -> torch.Tensor:
        """Build the soft Bellman target, replaying the actor's hidden state
        one step forward from `batch.h_actor_prev` (the state entering
        `batch.observations`) to get the state entering `batch.next_observations`.
        """
        with torch.no_grad():
            _, _, _, h_next = self.actor.get_action(batch.observations, batch.h_actor_prev)
            next_actions, next_log_pi, _, _ = self.actor.get_action(batch.next_observations, h_next)
            qf1_next_target = self.qf1_target(batch.next_observations, next_actions)
            qf2_next_target = self.qf2_target(batch.next_observations, next_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.current_alpha * next_log_pi
            return batch.rewards.flatten() + (1.0 - batch.dones.flatten()) * self.gamma * min_qf_next_target.view(-1)

    def compute_actor_loss(self, batch: RecurrentSACBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the actor loss and the sampled log-probability, replaying
        the actor from `batch.h_actor_prev` (the hidden state that was
        actually present when `batch.observations` was recorded).
        """
        pi, log_pi, _, _ = self.actor.get_action(batch.observations, batch.h_actor_prev)
        qf1_pi = self.qf1(batch.observations, pi)
        qf2_pi = self.qf2(batch.observations, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.current_alpha * log_pi) - min_qf_pi).mean()
        return actor_loss, log_pi

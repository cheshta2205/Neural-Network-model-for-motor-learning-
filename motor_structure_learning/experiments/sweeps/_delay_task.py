"""Shared building blocks for the instructed-delay motor task variant.

Adds a single binary go-cue to the observation (0 during a randomized delay
period, 1 once movement is permitted) so the task stays solvable while
preventing the agent from anticipating movement onset. Everything else
(networks, trainers, EC/CB, checkpoint format) is reused unchanged from
`_mc_sac_impl` / `sweeps.common` -- see `feedforward_delay.py`/`recurrent_delay.py`
for the episode loops that actually use these.
"""

from __future__ import annotations

import numpy as np

from motor_structure_learning.experiments import _mc_sac_impl as base
from motor_structure_learning.experiments.sweeps import common as sweep

DELAY_OBS_DIM = base.OBS_DIM + 1

HAZARD_P = 0.1
MAX_DELAY = 30
PREMATURE_ACTION_THRESHOLD = 0.1  # tolerance zone: no penalty below this action norm
PREMATURE_PENALTY = 5 * base.SUCCESS_BONUS  # coefficient: penalty = PREMATURE_PENALTY * excess norm


def build_obs_delay(pos: np.ndarray, goal: np.ndarray, prev_action: np.ndarray, go_cue: float) -> np.ndarray:
    """Base observation (pos, goal, prev_action) plus a binary go-cue: 0 during
    the delay period, 1 once movement is permitted.
    """
    return np.concatenate([pos, goal, prev_action, [go_cue]], axis=0).astype(np.float32)


def sample_delay_length(rng: np.random.Generator, hazard_p: float = HAZARD_P, max_delay: int = MAX_DELAY) -> int:
    """Sample a delay length (in steps) from a truncated geometric distribution:
    a constant per-step probability `hazard_p` of the go-cue firing, capped at
    `max_delay` so no trial runs pathologically long. Equivalent to simulating
    the hazard step-by-step, but drawn in one shot -- `rng.geometric` already
    returns "steps until first success" (support {1, 2, ...}), so subtracting
    1 gives the number of no-go steps before it (support {0, 1, ...}).
    """
    delay = int(rng.geometric(hazard_p)) - 1
    return min(delay, max_delay)


def compute_aug_obs_dim_delay(use_ec: bool, use_cb: bool, use_oracle_ctx: bool) -> int:
    """Same as `sweep._compute_aug_obs_dim`, plus the one extra go-cue dim."""
    return sweep._compute_aug_obs_dim(use_ec, use_cb, use_oracle_ctx) + 1

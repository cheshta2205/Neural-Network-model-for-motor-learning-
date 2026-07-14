"""Backward-compatible data API.

Prefer importing from motor_structure_learning.core.data in new code.
"""

from motor_structure_learning.core.data import simulate_state_action_and_next_state

__all__ = ["simulate_state_action_and_next_state"]
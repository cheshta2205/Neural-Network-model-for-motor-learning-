"""Core APIs for models and data used across experiments.

This package currently re-exports existing top-level modules so new code can
use a stable namespace while legacy scripts keep working.
"""

from .data import simulate_state_action_and_next_state
from .ec import EntorhinalCortex
from .hc import Hippocampus
from .models import CerebellumForwardModelContinuous

__all__ = [
    "simulate_state_action_and_next_state",
    "CerebellumForwardModelContinuous",
    "EntorhinalCortex",
    "Hippocampus",
]

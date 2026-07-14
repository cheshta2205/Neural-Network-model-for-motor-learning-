"""Backward-compatible HC module shim.

Prefer importing from motor_structure_learning.core.hc in new code.
"""

from motor_structure_learning.core.hc import Hippocampus

__all__ = ["Hippocampus"]

"""Backward-compatible CB module shim.

Prefer importing from motor_structure_learning.core.models in new code.
"""

from motor_structure_learning.core.models import CerebellumForwardModelContinuous

__all__ = ["CerebellumForwardModelContinuous"]

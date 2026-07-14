"""Backward-compatible training entrypoint.

Prefer importing from motor_structure_learning.experiments.train.
"""

from motor_structure_learning.experiments.train import train


if __name__ == "__main__":
    train()
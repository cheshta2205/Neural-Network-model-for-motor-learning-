"""Backward-compatible savings-all entrypoint.

Prefer importing from motor_structure_learning.experiments.mc_savings_all.
"""

from motor_structure_learning.experiments.mc_savings_all import main


if __name__ == "__main__":
    main()
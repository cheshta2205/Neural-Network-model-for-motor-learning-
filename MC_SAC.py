"""Backward-compatible SAC experiment entrypoint.

Prefer importing from motor_structure_learning.experiments.mc_sac.
"""

from motor_structure_learning.experiments.mc_sac import main


if __name__ == "__main__":
    main()
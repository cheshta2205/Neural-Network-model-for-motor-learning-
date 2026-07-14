"""Backward-compatible replay experiment entrypoint.

Prefer importing from motor_structure_learning.experiments.mc_replay_hc.
"""

from motor_structure_learning.experiments.mc_replay_hc import main


if __name__ == "__main__":
    main()
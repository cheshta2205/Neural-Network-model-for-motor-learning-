"""Backward-compatible SAC experiment entrypoint.

Prefer importing from motor_structure_learning.experiments.mc_sac_ec_cb.
"""

from motor_structure_learning.experiments.mc_sac_ec_cb import main


if __name__ == "__main__":
    main()
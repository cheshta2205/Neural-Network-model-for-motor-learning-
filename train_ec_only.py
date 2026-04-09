"""Backward-compatible EC-only training entrypoint.

Prefer importing from motor_structure_learning.experiments.train_ec_only.
"""

from motor_structure_learning.experiments.train_ec_only import main


if __name__ == "__main__":
    main()

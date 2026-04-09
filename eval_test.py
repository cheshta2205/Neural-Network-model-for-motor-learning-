"""Backward-compatible eval entrypoint.

Prefer importing from motor_structure_learning.experiments.eval_test.
"""

from motor_structure_learning.experiments.eval_test import main


if __name__ == "__main__":
    main()
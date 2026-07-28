"""Backward-compatible OLM task entrypoint.

Prefer importing from motor_structure_learning.experiments.olm_task.
"""

from motor_structure_learning.experiments.olm_task import main


if __name__ == "__main__":
    main()

"""Backward-compatible OLM task entrypoint.

Prefer importing from motor_structure_learning.experiments.olm_task.

Defaults to --kind closed_loop (train_olm_closed_loop_multi_seed, 5 seeds,
checkpoints/olm_closed_loop_multi_seed) -- pass --kind standard for the standard/
open-loop model instead (train_olm_multi_seed, checkpoints/olm_multi_seed).

Run directly: python olm_task.py [--kind {closed_loop,standard}]
"""

import argparse

from motor_structure_learning.experiments.olm_task import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("closed_loop", "standard"), default="closed_loop",
                         help="closed_loop: train_olm_closed_loop_multi_seed, "
                              "checkpoints/olm_closed_loop_multi_seed. standard: train_olm_multi_seed, "
                              "checkpoints/olm_multi_seed. (default: %(default)s)")
    args = parser.parse_args()

    main(kind=args.kind)

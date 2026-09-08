"""Package entrypoint for the Object Location Memory (OLM) task.

Always trains 5 seeds. Defaults to kind="closed_loop" (train_olm_closed_loop_multi_seed,
checkpoints/olm_closed_loop_multi_seed) -- that's the primary model going forward. Pass
kind="standard" for the standard/open-loop model instead (train_olm_multi_seed,
checkpoints/olm_multi_seed) -- see _olm_task_impl.main.
"""


def main(**kwargs):
    from . import _olm_task_impl as impl

    return impl.main(seeds=5, **kwargs)


if __name__ == "__main__":
    main()

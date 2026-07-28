"""Package entrypoint for the Object Location Memory (OLM) task."""


def main(**kwargs):
    from . import _olm_task_impl as impl

    return impl.main(seeds=5, **kwargs)


if __name__ == "__main__":
    main()

"""Package entrypoint for the replay-heavy MC ablation experiment."""


def main(**kwargs):
    from . import _mc_replay_hc_impl as impl

    return impl.main(**kwargs)


if __name__ == "__main__":
    main()
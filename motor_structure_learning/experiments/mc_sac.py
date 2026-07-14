"""Package entrypoint for the motor SAC experiment."""


def main(**kwargs):
    from . import _mc_sac_impl as impl

    return impl.main(**kwargs)


if __name__ == "__main__":
    main()
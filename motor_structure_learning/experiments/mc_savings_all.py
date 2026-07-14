"""Package entrypoint for the savings-all experiment."""


def main():
    from . import _mc_savings_all_impl as impl

    return impl.main()


if __name__ == "__main__":
    main()
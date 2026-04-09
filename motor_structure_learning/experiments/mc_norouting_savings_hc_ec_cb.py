"""Package entrypoint for no-routing HC+EC+CB savings experiment."""


def main():
    from . import _mc_norouting_savings_hc_ec_cb_impl as impl

    return impl.main()


if __name__ == "__main__":
    main()
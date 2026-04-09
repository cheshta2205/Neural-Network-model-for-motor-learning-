"""Package entrypoint for no-routing HC PPO savings experiment."""


def main():
    from . import _mc_norouting_savings_hc_ppo_impl as impl

    return impl.main()


if __name__ == "__main__":
    main()
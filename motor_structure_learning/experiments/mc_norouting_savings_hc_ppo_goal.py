"""Package entrypoint for goal-conditioned no-routing HC PPO savings."""


def main():
    from MC_norouting_savings_HC_PPO_goal import main as root_main

    return root_main()


if __name__ == "__main__":
    main()
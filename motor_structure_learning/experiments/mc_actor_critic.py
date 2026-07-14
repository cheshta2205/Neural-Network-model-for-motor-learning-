"""Package entrypoint for actor-critic learning experiment."""


def main():
    from MC_ActorCritic import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

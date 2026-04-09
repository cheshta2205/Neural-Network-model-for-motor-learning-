"""Package entrypoint for RL experiment."""


def main():
    from MC_RL import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

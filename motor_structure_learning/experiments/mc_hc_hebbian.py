"""Package entrypoint for HC Hebbian learning experiment."""


def main():
    from MC_HC_hebbian import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

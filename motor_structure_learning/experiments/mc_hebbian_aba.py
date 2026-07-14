"""Package entrypoint for Hebbian ABA learning experiment."""


def main():
    from mc_hebbian_ABA import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

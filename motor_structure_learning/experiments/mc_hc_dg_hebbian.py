"""Package entrypoint for HC-DG Hebbian learning experiment."""


def main():
    from MC_HC_DG_hebbian import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

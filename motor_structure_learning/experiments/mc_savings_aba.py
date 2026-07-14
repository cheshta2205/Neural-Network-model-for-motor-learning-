"""Package entrypoint for ABA savings experiment."""


def main():
    from MC_savings_ABA import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

"""Package entrypoint for forgetting experiment."""


def main():
    from MC_forgetting import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

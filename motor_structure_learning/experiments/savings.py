"""Package entrypoint for core savings experiment."""


def main():
    from savings import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

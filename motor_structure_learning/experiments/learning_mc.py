"""Package entrypoint for basic motor learning experiment."""


def main():
    from learning_MC import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

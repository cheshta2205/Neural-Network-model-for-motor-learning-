"""Package entrypoint for continuous motor learning experiment."""


def main():
    from learning_MC_continuous import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

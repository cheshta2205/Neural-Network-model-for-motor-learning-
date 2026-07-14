"""Package entrypoint for new-diagonal EC experiment."""


def main():
    from learning_newDiagEC import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

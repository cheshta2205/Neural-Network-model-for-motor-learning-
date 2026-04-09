"""Package entrypoint for CBonly novel learning experiment."""


def main():
    from learning_MC_CBonly_novel import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

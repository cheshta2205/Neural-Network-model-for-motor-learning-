"""Package entrypoint for 4-condition savings experiment."""


def main():
    from MC_savings_4cond import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

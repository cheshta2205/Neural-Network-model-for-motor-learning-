"""Package entrypoint for routing savings experiment."""


def main():
    from MC_routing_savings import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

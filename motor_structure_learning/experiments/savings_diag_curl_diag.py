"""Package entrypoint for savings diagonal experiment."""


def main():
    from savings_diag_curl_diag import main as root_main

    return root_main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python

from sys import argv

from autofit.tools import edenise


def main():
    try:
        root_directory, name, prefix = argv[1:]
        edenise.edenise(
            root_directory,
            name,
            prefix
        )
    except ValueError:
        print("Usage: ./edenise.py root_directory project_name import_prefix")
        print("e.g.: ./edenise.py /path/to/autofit autofit af")
        exit(1)


if __name__ == "__main__":
    main()

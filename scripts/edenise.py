#!/usr/bin/env python

from configparser import ConfigParser
from sys import argv

from autofit.tools import edenise


def main(
        root_directory
):
    try:
        config = ConfigParser()
        config.read(
            f"{root_directory}/eden.ini"
        )

        edenise.edenise(
            root_directory,
            config.get("eden", "name"),
            config.get("eden", "prefix")
        )
    except ValueError:
        print("Usage: ./edenise.py root_directory")
        exit(1)


if __name__ == "__main__":
    main(
        argv[1]
    )

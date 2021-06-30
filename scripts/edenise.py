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

        eden_dependencies = [
            dependency.strip()
            for dependency
            in config.get(
                "eden",
                "eden_dependencies"
            ).split(",")
        ]

        edenise.edenise(
            root_directory,
            config.get("eden", "name"),
            config.get("eden", "prefix"),
            config.get("eden", "eden_prefix"),
            eden_dependencies,
            config.get("eden", "should_rename_modules").lower().startswith("t"),
            config.get("eden", "should_remove_type_annotations").lower().startswith("t"),
        )
    except ValueError:
        print("Usage: ./edenise.py root_directory")
        exit(1)


if __name__ == "__main__":
    main(
        argv[1]
    )

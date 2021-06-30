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
            root_directory=root_directory,
            name=config.get("eden", "name"),
            prefix=config.get("eden", "prefix"),
            eden_prefix=config.get("eden", "eden_prefix"),
            eden_dependencies=eden_dependencies,
            should_rename_modules=config.get("eden", "should_rename_modules").lower().startswith("t"),
            should_remove_type_annotations=config.get("eden", "should_remove_type_annotations").lower().startswith("t"),
        )
    except ValueError:
        print("Usage: ./edenise.py root_directory")
        exit(1)


if __name__ == "__main__":
    main(
        argv[1]
    )

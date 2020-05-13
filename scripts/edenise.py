#!/usr/bin/env python

from os import path
from autofit.tools import edenise


def main():
    root_directory = f"{path.dirname(path.realpath(__file__))}/.."

    name = "autofit"
    prefix = "af"
    edenise.edenise(
        root_directory,
        name,
        prefix
    )


if __name__ == "__main__":
    main()

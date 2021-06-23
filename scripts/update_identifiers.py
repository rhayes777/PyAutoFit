#!/usr/bin/env python

"""
Usage:

./update_identifiers.py path/to/output
"""
from sys import argv

from autofit.tools.update_identifiers import update_identifiers

if __name__ == "__main__":
    update_identifiers(
        argv[1],
        keep_source_directory=True
    )

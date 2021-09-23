#!/usr/bin/env python

"""
Update the identifier for searches in a directory by mapping values in
the identifier file to new values.

This should be applied when the source code has changed meaning that pickles
cannot be loaded, for example when my.path.Class moves to my.Class.

Usage:
./update_identifiers_from_file.py directory map_file.yaml

map_file.yaml must be a yaml file mapping old values to new.

e.g.
old: new
something: else
"""

import logging
from sys import argv

from autofit.tools.update_identifiers import update_identifiers_from_file

if __name__ == "__main__":
    try:
        directory, map_file = argv[1:]
        update_identifiers_from_file(
            output_directory=directory,
            map_filename=map_file
        )
    except Exception as e:
        logging.exception(e)
        print(
            "Usage: update_identifiers_from_file.py directory map_file.yaml"
        )

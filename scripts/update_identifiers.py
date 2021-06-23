#!/usr/bin/env python

"""
Update identifiers for previous searches to version defined in config.

If a directory is passed, then the operation is performed on files in the directory.
If the path to a database file is passed, then the operation is performed on that
database.

Usage:

./update_identifiers.py path/to/output
./update_identifiers.py path/to/database.sqlite
"""
from sys import argv

import autofit as af
from autofit.tools.update_identifiers import update_directory_identifiers, update_database_identifiers

if __name__ == "__main__":
    argument = argv[1]

    if argument.endswith(
            ".sqlite"
    ):
        update_database_identifiers(
            af.database.open_database(
                argument
            )
        )
    else:
        update_directory_identifiers(
            argument
        )

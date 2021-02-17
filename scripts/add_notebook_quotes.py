#!/usr/bin/env python
"""
Usage
./add_notebook_quotes.py /path/to/input /path/to/output
"""

from sys import argv

from autofit.tools.add_notebook_quotes import add_notebook_quotes

if __name__ == "__main__":
    _, in_filename, out_filename = argv

    with open(in_filename) as f:
        lines = f.readlines()

    with open(out_filename, "w+") as f:
        f.writelines(
            add_notebook_quotes(
                lines
            )
        )
#!/usr/bin/env python
from sys import argv

import autofit as af

if __name__ == "__main__":
    aggregator = af.Aggregator(
        af.db.open_database(
            "sqlite://"
        )
    )
    aggregator.add_directory(
        argv[1]
    )
    print(aggregator)

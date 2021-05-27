from pathlib import Path

import autofit as af


def test():
    af.Aggregator.from_database(
        Path(
            __file__
        ).parent / "database.sqlite"
    )

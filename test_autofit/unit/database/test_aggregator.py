from pathlib import Path

from autofit.database.aggregator import Aggregator

directory = Path(__file__).parent


def _test_load(session):
    aggregator = Aggregator(
        session
    )
    aggregator.add_directory(
        str(directory)
    )

    assert len(aggregator) == 1

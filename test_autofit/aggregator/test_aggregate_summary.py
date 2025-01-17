from autofit.aggregator import Aggregator
from pathlib import Path

from autofit.aggregator.aggregate_summary import AggregateSummary

import pytest


@pytest.fixture
def output_path():
    path = Path("/tmp/summary.csv")
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def summary():
    directory = Path(__file__).parent / "aggregate_summary"
    aggregator = Aggregator.from_directory(directory)
    return AggregateSummary(aggregator)


def test_writes(output_path, summary):
    summary.save(output_path)

    assert output_path.exists()

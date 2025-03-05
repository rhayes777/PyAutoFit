import pytest
from pathlib import Path
from autofit.aggregator import Aggregator


@pytest.fixture
def aggregator():
    directory = Path(__file__).parent / "aggregate_summary"
    return Aggregator.from_directory(directory)

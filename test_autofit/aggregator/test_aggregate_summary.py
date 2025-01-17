from autofit.aggregator import Aggregator
from pathlib import Path


def test():
    directory = Path(__file__).parent / "aggregate_summary"
    aggregator = Aggregator.from_directory(directory)
    print(len(aggregator))

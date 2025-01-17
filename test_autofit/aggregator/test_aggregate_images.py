import pytest
from pathlib import Path

from autofit.aggregator import Aggregator
from autofit.aggregator.aggregate_images import AggregateImages, Subplot


@pytest.fixture
def aggregate():
    directory = Path(__file__).parent / "aggregate_summary"
    aggregator = Aggregator.from_directory(directory)
    return AggregateImages(aggregator)


def test(aggregate):
    result = aggregate.extract_image(
        [
            Subplot.Data,
            Subplot.SourcePlaneZoomed,
        ]
    )
    assert result.size == (122, 120)

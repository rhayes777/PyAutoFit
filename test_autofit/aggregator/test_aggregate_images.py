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
        Subplot.Data,
        Subplot.SourcePlaneZoomed,
    )
    assert result.size == (122, 120)
    assert result == aggregate.extract_image(
        Subplot.Data,
        Subplot.SourcePlaneZoomed,
    )


def test_different_plots(aggregate):
    assert aggregate.extract_image(
        Subplot.Data,
        Subplot.SourcePlaneZoomed,
    ) != aggregate.extract_image(
        Subplot.SourcePlaneZoomed,
        Subplot.Data,
    )


def test_longer(aggregate):
    result = aggregate.extract_image(
        Subplot.NormalizedResidualMap,
        Subplot.SourcePlaneNoZoom,
        Subplot.SourceModelImage,
    )

    assert result.size == (183, 120)


def test_subplot_width(aggregate):
    result = aggregate.extract_image(
        Subplot.NormalizedResidualMap,
        Subplot.SourcePlaneNoZoom,
        Subplot.SourceModelImage,
        subplot_width=2,
    )

    assert result.size == (122, 240)


def test_output_to_folder(aggregate, output_directory):
    aggregate.output_to_folder(
        output_directory,
        Subplot.Data,
        Subplot.SourcePlaneZoomed,
        Subplot.SourceModelImage,
    )
    assert list(Path(output_directory).glob("*.png"))

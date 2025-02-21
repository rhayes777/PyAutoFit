import pytest
from pathlib import Path

from PIL import Image

from autofit.aggregator import Aggregator
from autofit.aggregator.aggregate_images import AggregateImages, Subplot


@pytest.fixture
def aggregator():
    directory = Path(__file__).parent / "aggregate_summary"
    return Aggregator.from_directory(directory)


@pytest.fixture
def aggregate(aggregator):
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


def test_output_to_folder_name(
    aggregate,
    output_directory,
    aggregator,
):
    aggregate.output_to_folder(
        output_directory,
        Subplot.Data,
        Subplot.SourcePlaneZoomed,
        Subplot.SourceModelImage,
        name="id",
    )

    id_ = next(iter(aggregator)).id
    assert list(Path(output_directory).glob(f"{id_}.png"))


def test_custom_images(
    aggregate,
    aggregator,
):
    image = Image.new("RGB", (10, 10), "white")
    images = [image for _ in aggregator]

    result = aggregate.extract_image(
        Subplot.Data,
        Subplot.SourcePlaneZoomed,
        Subplot.SourceModelImage,
        images,
    )

    result.show()

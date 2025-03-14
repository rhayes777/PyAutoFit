from enum import Enum

import pytest
from pathlib import Path

from PIL import Image

from autofit.aggregator import Aggregator
from autofit.aggregator.summary.aggregate_images import AggregateImages, SubplotFit


@pytest.fixture
def aggregate(aggregator):
    return AggregateImages(aggregator)


def test(aggregate):
    result = aggregate.extract_image(
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
        ],
    )
    assert result.size == (122, 120)
    assert result == aggregate.extract_image(
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
        ],
    )


def test_different_plots(aggregate):
    assert aggregate.extract_image(
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
        ],
    ) != aggregate.extract_image(
        [
            SubplotFit.SourcePlaneZoomed,
            SubplotFit.Data,
        ],
    )


def test_longer(aggregate):
    result = aggregate.extract_image(
        [
            SubplotFit.NormalizedResidualMap,
            SubplotFit.SourcePlaneNoZoom,
            SubplotFit.SourceModelImage,
        ],
    )

    assert result.size == (183, 120)


def test_subplot_width(aggregate):
    result = aggregate.extract_image(
        [
            SubplotFit.NormalizedResidualMap,
            SubplotFit.SourcePlaneNoZoom,
            SubplotFit.SourceModelImage,
        ],
        subplot_width=2,
    )

    assert result.size == (122, 240)


def test_output_to_folder(aggregate, output_directory):
    aggregate.output_to_folder(
        output_directory,
        "name",
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
            SubplotFit.SourceModelImage,
        ],
    )
    assert list(Path(output_directory).glob("*.png"))


def test_list_of_names(aggregate, output_directory):
    aggregate.output_to_folder(
        output_directory,
        ["one", "two"],
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
            SubplotFit.SourceModelImage,
        ],
    )
    assert [path.name for path in Path(output_directory).glob("*.png")] == [
        "two.png",
        "one.png",
    ]


def test_output_to_folder_name(
    aggregate,
    output_directory,
    aggregator,
):
    aggregate.output_to_folder(
        output_directory,
        "id",
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
            SubplotFit.SourceModelImage,
        ],
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
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
            SubplotFit.SourceModelImage,
            images,
        ]
    )

    assert result.size == (193, 120)


def test_custom_function(aggregate):
    def make_image(output):
        return Image.new("RGB", (10, 10), "white")

    result = aggregate.extract_image(
        [
            SubplotFit.Data,
            SubplotFit.SourcePlaneZoomed,
            SubplotFit.SourceModelImage,
            make_image,
        ]
    )

    assert result.size == (193, 120)


def test_custom_subplot_fit(aggregate):
    class SubplotFit(Enum):
        """
        The subplots that can be extracted from the subplot_fit image.

        The values correspond to the position of the subplot in the 4x3 grid.
        """

        Data = (0, 0)

    result = aggregate.extract_image(
        [
            SubplotFit.Data,
        ]
    )
    assert result.size == (61, 120)


def test_bad_aggregator():
    directory = Path(__file__).parent / "aggregate_summaries"
    aggregator = Aggregator.from_directory(directory)

    with pytest.raises(ValueError):
        AggregateImages(aggregator)

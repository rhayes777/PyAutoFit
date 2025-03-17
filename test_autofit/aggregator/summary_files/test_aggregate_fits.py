import pytest

import autofit as af
from pathlib import Path


@pytest.fixture(name="summary")
def make_summary(aggregator):
    return af.AggregateFITS(aggregator)


def test_aggregate(summary):
    result = summary.extract_fits(
        [
            af.FitFITS.ModelImage,
            af.FitFITS.ResidualMap,
        ],
    )
    assert len(result) == 5


def test_output_to_file(summary, output_directory):
    folder = output_directory / "fits"
    summary.output_to_folder(
        folder,
        name="id",
        hdus=[
            af.FitFITS.ModelImage,
            af.FitFITS.ResidualMap,
        ],
    )
    assert list(folder.glob("*"))


def test_list_of_names(summary, output_directory):
    summary.output_to_folder(
        output_directory,
        ["one", "two"],
        [
            af.FitFITS.ModelImage,
            af.FitFITS.ResidualMap,
        ],
    )
    assert [path.name for path in Path(output_directory).glob("*.fits")] == [
        "one.fits",
        "two.fits",
    ]

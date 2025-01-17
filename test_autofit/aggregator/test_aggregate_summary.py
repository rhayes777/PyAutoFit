import csv

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

    with open(output_path) as f:
        dicts = list(csv.DictReader(f))

    assert dicts[0]["id"] is not None
    assert dicts[1]["id"] is not None


def test_add_column(output_path, summary):
    summary.add_column("galaxies.lens.bulge.centre.centre_0")
    summary.save(output_path)

    with open(output_path) as f:
        dicts = list(csv.DictReader(f))

    assert dicts[0]["galaxies_lens_bulge_centre_centre_0"] == "-1.0"
    assert dicts[1]["galaxies_lens_bulge_centre_centre_0"] == "-5.0"


def test_add_named_column(output_path, summary):
    summary.add_column(
        "galaxies.lens.bulge.centre.centre_0",
        name="centre_0",
    )
    summary.save(output_path)

    with open(output_path) as f:
        dicts = list(csv.DictReader(f))

    assert dicts[0]["centre_0"] == "-1.0"
    assert dicts[1]["centre_0"] == "-5.0"

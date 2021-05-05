from os import path

import pytest

import autofit as af
from autofit.mock.mock import MockSearchOutput


@pytest.fixture(name="path_aggregator")
def make_path_aggregator(session):
    fits = [
        af.db.Fit(
            id="complete",
            is_complete=True
        ),
        af.db.Fit(
            id="incomplete",
            is_complete=False
        )
    ]
    for fit in fits:
        fit["dataset"] = {
            "name": "dataset"
        }
    session.add_all(fits)
    session.flush()
    return af.Aggregator(session)


@pytest.fixture(name="aggregator_directory")
def make_aggregator_directory():
    directory = path.dirname(path.realpath(__file__))

    return path.join(f"{directory}", "..", "tools", "files", "aggregator")


@pytest.fixture(name="aggregator")
def make_aggregator():
    aggregator = af.Aggregator("")
    aggregator.search_outputs = [
        MockSearchOutput(path.join("directory", "number", "one"), "pipeline1", "search1", "dataset1"),
        MockSearchOutput(path.join("directory", "number", "two"), "pipeline1", "search2", "dataset1"),
        MockSearchOutput(path.join("directory", "letter", "a"), "pipeline2", "search2", "dataset2"),
    ]
    return aggregator

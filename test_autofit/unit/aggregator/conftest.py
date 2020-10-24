import os

import pytest

import autofit as af
from autofit.mock.mock import MockPhaseOutput


@pytest.fixture(
    name="path_aggregator"
)
def make_path_aggregator(
        aggregator_directory
):
    aggregator = af.Aggregator(
        aggregator_directory
    )
    yield aggregator
    aggregator.remove_unzipped()


@pytest.fixture(
    name="aggregator_directory"
)
def make_aggregator_directory():
    directory = os.path.dirname(
        os.path.realpath(__file__)
    )
    return f"{directory}/../tools/files/aggregator"


@pytest.fixture(name="aggregator")
def make_aggregator():
    aggregator = af.Aggregator("")
    aggregator.phases = [
        MockPhaseOutput(
            "directory/number/one",
            "pipeline1",
            "phase1",
            "dataset1"
        ),
        MockPhaseOutput(
            "directory/number/two",
            "pipeline1",
            "phase2",
            "dataset1"
        ),
        MockPhaseOutput(
            "directory/letter/a",
            "pipeline2",
            "phase2",
            "dataset2"
        ),
    ]
    return aggregator

import os

import pytest

import autofit as af


class MedianPDFInstance:
    def __init__(self, name):
        self.name = name


class MockPhaseOutput:
    def __init__(self, directory, pipeline, phase, dataset):
        self.directory = directory
        self.pipeline = pipeline
        self.phase = phase
        self.dataset = dataset

    @property
    def median_pdf_instance(self):
        return MedianPDFInstance(
            self.phase
        )

    @property
    def output(self):
        return self


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

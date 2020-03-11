import pytest

import autofit as af


class MockPhaseOutput:
    def __init__(self, directory, pipeline, phase, dataset):
        self.directory = directory
        self.pipeline = pipeline
        self.phase = phase
        self.dataset = dataset


@pytest.fixture(name="aggregator")
def make_aggregator():
    aggregator = af.Aggregator("")
    aggregator.phases = [
        MockPhaseOutput("directory/number/one", "pipeline1", "phase1", "dataset1"),
        MockPhaseOutput("directory/number/two", "pipeline1", "phase2", "dataset1"),
        MockPhaseOutput("directory/letter/a", "pipeline2", "phase2", "dataset2"),
    ]
    return aggregator


def test_attribute(aggregator):
    assert aggregator.pipeline == ["pipeline1", "pipeline1", "pipeline2"]
    assert aggregator.phase == ["phase1", "phase2", "phase2"]
    assert aggregator.dataset == ["dataset1", "dataset1", "dataset2"]


def test_indexing(aggregator):
    assert aggregator[1:].pipeline == ["pipeline1", "pipeline2"]
    assert aggregator[-1:].pipeline == ["pipeline2"]
    assert aggregator[0].pipeline == "pipeline1"


def test_filter_index(aggregator):
    assert aggregator.filter(
        pipeline="pipeline1"
    )[1:].pipeline == ["pipeline1"]
    assert aggregator[1:].filter(
        pipeline="pipeline1"
    ).pipeline == ["pipeline1"]


def test_filter(aggregator):
    result = aggregator.filter(pipeline="pipeline1")
    assert len(result) == 2
    assert result[0].pipeline == "pipeline1"

    result = aggregator.filter(pipeline="pipeline1", phase="phase1")

    assert len(result) == 1
    assert result[0].pipeline == "pipeline1"

    result = aggregator.filter(pipeline="pipeline1").filter(phase="phase1")

    assert len(result) == 1
    assert result[0].pipeline == "pipeline1"


def test_filter_contains(aggregator):
    result = aggregator.filter_contains(pipeline="1")
    assert len(result) == 2
    assert result[0].pipeline == "pipeline1"

    result = aggregator.filter_contains(pipeline="1", phase="1")

    assert len(result) == 1
    assert result[0].pipeline == "pipeline1"

    result = aggregator.filter_contains(pipeline="1").filter(phase="phase1")

    assert len(result) == 1
    assert result[0].pipeline == "pipeline1"


def test_filter_contains_directory(aggregator):
    result = aggregator.filter_contains(directory="number")
    assert len(result) == 2

    result = aggregator.filter_contains(directory="letter")
    assert len(result) == 1


def test_group_by(aggregator):
    result = aggregator.group_by("pipeline")

    assert len(result) == 2
    assert len(result[0]) == 2
    assert len(result[1]) == 1

    result = result.filter(phase="phase2")

    assert len(result) == 2
    assert len(result[0]) == 1
    assert len(result[1]) == 1

    assert result.phase == [["phase2"], ["phase2"]]

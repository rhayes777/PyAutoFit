import os

import pytest

import autofit as af


class MostProbableInstance:
    def __init__(self, name):
        self.name = name


class MockPhaseOutput:
    def __init__(self, directory, pipeline, phase, dataset):
        self.directory = directory
        self.pipeline = pipeline
        self.phase = phase
        self.dataset = dataset

    @property
    def most_probable_instance(self):
        return MostProbableInstance(
            self.phase
        )

    @property
    def output(self):
        return self


@pytest.fixture(name="aggregator")
def make_aggregator():
    aggregator = af.Aggregator("")
    aggregator.phases = [
        MockPhaseOutput("directory/number/one", "pipeline1", "phase1", "dataset1"),
        MockPhaseOutput("directory/number/two", "pipeline1", "phase2", "dataset1"),
        MockPhaseOutput("directory/letter/a", "pipeline2", "phase2", "dataset2"),
    ]
    return aggregator


@pytest.fixture(
    name="aggregator_directory"
)
def make_aggregator_directory():
    directory = os.path.dirname(
        os.path.realpath(__file__)
    )
    return f"{directory}/test_files/aggregator"


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


def test_completed_aggregator(
        aggregator_directory
):
    aggregator = af.Aggregator(
        aggregator_directory,
        completed_only=True
    )
    assert len(aggregator) == 1
    assert "completed" in aggregator[0].directory


class TestLoading:
    def test_unzip(self, path_aggregator):
        assert len(path_aggregator) == 2

    def test_pickles(self, path_aggregator):
        assert list(path_aggregator.values(
            "dataset"
        ))[0]["name"] == "dataset"
        assert list(path_aggregator.values(
            "model"
        ))[0]["name"] == "model"
        assert list(path_aggregator.values(
            "optimizer"
        ))[0]["name"] == "optimizer"


class TestOperations:
    def test_not_contains(self, aggregator):
        predicate = ~(aggregator.pipeline.contains("1"))
        result = aggregator.filter(
            predicate
        )
        assert len(result) == 1
        assert result[0].pipeline == "pipeline2"

    def test_not_equal(self, aggregator):
        predicate = aggregator.pipeline != "pipeline1"
        result = aggregator.filter(
            predicate
        )
        assert len(result) == 1
        assert result[0].pipeline == "pipeline2"

    def test_rhs(self, aggregator):
        predicate = "pipeline1" != aggregator.pipeline
        result = aggregator.filter(
            predicate
        )
        assert result.pipeline == ["pipeline2"]

        predicate = "pipeline1" == aggregator.pipeline
        result = aggregator.filter(
            predicate
        )
        assert result.pipeline == ["pipeline1", "pipeline1"]

    def test_attribute(self, aggregator):
        assert list(
            aggregator.values("pipeline")
        ) == ["pipeline1", "pipeline1", "pipeline2"]
        assert list(
            aggregator.values("phase")
        ) == ["phase1", "phase2", "phase2"]
        assert list(
            aggregator.values("dataset")
        ) == ["dataset1", "dataset1", "dataset2"]

    def test_indexing(self, aggregator):
        assert list(
            aggregator[1:].values("pipeline")
        ) == ["pipeline1", "pipeline2"]
        assert list(
            aggregator[-1:].values("pipeline")
        ) == ["pipeline2"]
        assert aggregator[0].pipeline == "pipeline1"

    def test_filter_index(self, aggregator):
        assert list(aggregator.filter(
            aggregator.pipeline == "pipeline1"
        )[1:].values("pipeline")) == ["pipeline1"]
        assert list(aggregator[1:].filter(
            aggregator.pipeline == "pipeline1"
        ).values("pipeline")) == ["pipeline1"]

    def test_filter(self, aggregator):
        result = aggregator.filter(
            aggregator.pipeline == "pipeline1"
        )
        assert len(result) == 2
        assert result[0].pipeline == "pipeline1"

        result = aggregator.filter(
            aggregator.pipeline == "pipeline1",
            aggregator.phase == "phase1"
        )

        assert len(result) == 1
        assert result[0].pipeline == "pipeline1"

        result = aggregator.filter(
            aggregator.pipeline == "pipeline1"
        ).filter(
            aggregator.phase == "phase1"
        )

        assert len(result) == 1
        assert result[0].pipeline == "pipeline1"

    def test_filter_contains(self, aggregator):
        result = aggregator.filter(
            aggregator.pipeline.contains("1")
        )
        assert len(result) == 2
        assert result[0].pipeline == "pipeline1"

        result = aggregator.filter(
            aggregator.pipeline.contains("1"),
            aggregator.phase.contains("1")
        )

        assert len(result) == 1
        assert result[0].pipeline == "pipeline1"

        result = aggregator.filter(
            aggregator.pipeline.contains("1")
        ).filter(
            aggregator.phase == "phase1"
        )

        assert len(result) == 1
        assert result[0].pipeline == "pipeline1"

    def test_filter_contains_directory(self, aggregator):
        result = aggregator.filter(
            aggregator.directory.contains("number")
        )
        assert len(result) == 2

        result = aggregator.filter(
            aggregator.directory.contains("letter")
        )
        assert len(result) == 1

    def test_group_by(self, aggregator):
        result = aggregator.group_by("pipeline")

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1

        result = result.filter(aggregator.phase == "phase2")

        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1

        assert list(map(
            list,
            result.values("phase")
        )) == [["phase2"], ["phase2"]]

    def test_map(self, aggregator):
        def some_function(output):
            return f"{output.phase} {output.dataset}"

        results = aggregator.map(some_function)
        assert list(results) == [
            "phase1 dataset1",
            "phase2 dataset1",
            "phase2 dataset2",
        ]

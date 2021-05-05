from os import path

import pytest

import autofit as af
from autofit.mock.mock import MockSearchOutput


def test_completed_aggregator(
        path_aggregator
):
    aggregator = path_aggregator(
        path_aggregator.is_complete
    )
    assert len(aggregator) == 1


class TestLoading:
    def test_unzip(self, path_aggregator):
        assert len(path_aggregator) == 2

    def test_pickles(self, path_aggregator):
        assert list(path_aggregator.values("dataset"))[0]["name"] == "dataset"


class TestOperations:
    def test_attribute(self, aggregator):
        assert list(aggregator.values("pipeline")) == [
            "pipeline1",
            "pipeline1",
            "pipeline2",
        ]
        assert list(aggregator.values("search")) == ["search1", "search2", "search2"]
        assert list(aggregator.values("dataset")) == [
            "dataset1",
            "dataset1",
            "dataset2",
        ]

    def test_indexing(self, aggregator):
        assert list(aggregator[1:].values("pipeline")) == ["pipeline1", "pipeline2"]
        assert list(aggregator[-1:].values("pipeline")) == ["pipeline2"]
        assert aggregator[0].pipeline == "pipeline1"

    def test_group_by(self, aggregator):
        result = aggregator.group_by("pipeline")

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1

        result = result.filter(aggregator.search == "search2")

        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1

        assert list(map(list, result.values("search"))) == [["search2"], ["search2"]]

    def test_map(self, aggregator):
        def some_function(output):
            return f"{output.search} {output.dataset}"

        results = aggregator.map(some_function)
        assert list(results) == [
            "search1 dataset1",
            "search2 dataset1",
            "search2 dataset2",
        ]

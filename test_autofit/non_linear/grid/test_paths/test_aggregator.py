import pytest

from autofit.aggregator.aggregator import Aggregator


@pytest.fixture(name="aggregator")
def make_aggregator(sample_name_paths, grid_search_10_result):
    return Aggregator.from_directory(sample_name_paths.output_path)


def test_aggregate(aggregator):
    assert len(aggregator) == 100
    assert len(aggregator.grid_search_outputs) == 1


def test_correspondence(aggregator):
    (grid_search,) = aggregator.grid_searches()

    assert len(grid_search.search_outputs) == 100
    assert grid_search.search_outputs[0] is list(aggregator)[0]

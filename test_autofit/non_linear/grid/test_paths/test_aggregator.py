from autofit.aggregator.aggregator import Aggregator


def test_aggregate(sample_name_paths, grid_search_10_result):
    aggregator = Aggregator.from_directory(sample_name_paths.output_path)
    assert len(aggregator) == 100
    assert len(aggregator.grid_search_outputs) == 1

def test_search_instance(
        grid_search
):
    assert grid_search.search_instance(
        grid_search.paths.identifier
    ).paths.output_path == grid_search.paths.output_path

def test_contain_identifier(
        grid_search,
        grid_paths
):
    for paths in grid_paths:
        identifier = grid_search.paths.identifier
        output_path = paths.output_path
        assert identifier in output_path
        assert not output_path.endswith(
            identifier
        )


def test_does_not_contain_identifier(
        grid_paths
):
    for paths in grid_paths:
        assert paths.identifier not in paths.output_path


def test_distinct_identifiers(
        grid_search,
        grid_paths
):
    identifiers = {
        paths.identifier
        for paths in grid_paths
    }
    assert len(identifiers) == 100
    assert grid_search.paths.identifier not in identifiers

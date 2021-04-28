import pytest

import autofit as af


def _make_grid_paths(
        grid_search,
        mapper
):
    jobs = list(
        grid_search.make_jobs(
            mapper,
            None,
            grid_priors=[
                mapper.component.one_tuple.one_tuple_0,
                mapper.component.one_tuple.one_tuple_1,
            ],
        )
    )
    return [
        job.search_instance.paths
        for job in jobs
    ]


@pytest.fixture(
    name="grid_paths"
)
def make_grid_paths(
        grid_search,
        mapper
):
    return _make_grid_paths(
        grid_search,
        mapper
    )


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


@pytest.fixture(
    name="database_paths"
)
def make_database_paths(
        grid_search,
        mapper,
        session
):
    grid_search.paths = af.DatabasePaths(
        session=session,
        name="grid_search"
    )

    return _make_grid_paths(
        grid_search,
        mapper
    )


def test_paths_type(
        database_paths
):
    for path in database_paths:
        assert isinstance(
            path,
            af.DatabasePaths
        )


# def test_parent(
#         database_paths,
#         mapper
# ):
#     paths = database_paths[0]
#     assert paths.fit.parent is not None
#     assert paths.fit.instance is not None
#     assert paths.fit.parent.is_grid_search

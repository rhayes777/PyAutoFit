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


@pytest.fixture(
    name="database_paths"
)
def make_database_paths(
        grid_search,
        mapper,
        session
):
    grid_search.search.paths = af.DatabasePaths(
        session=session,
        name="grid_search"
    )
    grid_search.search.paths.model = mapper

    return _make_grid_paths(
        grid_search,
        mapper
    )

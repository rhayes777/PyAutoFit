import pytest

import autofit as af
from autofit.mock.mock import MockAnalysis
from test_autofit.non_linear.grid.test_optimizer_grid_search import MockOptimizer


@pytest.fixture(
    name="parent_search"
)
def make_parent_search():
    return af.DynestyStatic()


@pytest.fixture(
    name="database_parent_search"
)
def make_database_parent_search(
        session
):
    return af.DynestyStatic(
        session=session
    )


def _make_grid_search(
        mapper,
        parent_search,
        session=None
):
    search = af.SearchGridSearch(
        search=MockOptimizer(
            session=session
        ),
        number_of_steps=2,
        parent_search=parent_search
    )
    search.fit(
        model=mapper,
        analysis=MockAnalysis(),
        grid_priors=[
            mapper.component.one_tuple.one_tuple_0,
            mapper.component.one_tuple.one_tuple_1,
        ]
    )
    return search


@pytest.fixture(
    name="grid_search"
)
def make_grid_search(
        mapper,
        parent_search
):
    return _make_grid_search(
        mapper,
        parent_search
    )


@pytest.fixture(
    name="database_grid_search"
)
def make_database_grid_search(
        mapper,
        database_parent_search,
        session
):
    return _make_grid_search(
        mapper,
        database_parent_search,
        session=session
    )


class TestDirectory:
    def test_parent_search(
            self,
            grid_search,
            parent_search
    ):
        assert parent_search.paths is grid_search.paths.parent

    def test_is_grid_search(
            self,
            grid_search
    ):
        assert grid_search.paths.is_grid_search


class TestDatabase:
    def test_parent_search(
            self,
            database_grid_search,
            database_parent_search
    ):
        assert database_parent_search.paths is database_grid_search.paths.parent

    def test_is_grid_search(
            self,
            database_grid_search
    ):
        assert database_grid_search.paths.is_grid_search

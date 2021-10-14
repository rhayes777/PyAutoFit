import pytest

import autofit as af
from autofit.mock.mock import MockAnalysis
from test_autofit.non_linear.grid.test_optimizer_grid_search import MockOptimizer


@pytest.fixture(
    name="parent_search"
)
def make_parent_search():
    return af.DynestyStatic(
        "parent"
    )


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
        number_of_steps=2
    )
    search.fit(
        model=mapper,
        analysis=MockAnalysis(),
        grid_priors=[
            mapper.component.one_tuple.one_tuple_0,
            mapper.component.one_tuple.one_tuple_1,
        ],
        parent=parent_search
    )
    return search


@pytest.fixture(
    name="grid_search"
)
def make_grid_search(
        mapper,
        parent_search
):
    search = _make_grid_search(
        mapper,
        parent_search
    )
    search.paths.save_all()
    return search


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


class TestMiscombination:
    def test_directory_for_database(
            self,
            parent_search,
            session,
            mapper
    ):
        with pytest.raises(TypeError):
            _make_grid_search(
                mapper,
                parent_search,
                session
            )

    def test_database_for_directory(
            self,
            grid_search,
            database_parent_search
    ):
        grid_paths = grid_search.paths
        parent_paths = database_parent_search.paths

        with open(
                grid_paths._parent_identifier_path
        ) as f:
            assert f.read() == parent_paths.identifier


class TestDirectory:
    def test_parent_search(
            self,
            grid_search,
            parent_search
    ):
        grid_paths = grid_search.paths
        parent_paths = parent_search.paths

        assert parent_paths is grid_paths.parent
        with open(
                grid_paths._parent_identifier_path
        ) as f:
            assert f.read() == parent_paths.identifier

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
        parent_paths = database_parent_search.paths

        assert parent_paths is database_grid_search.paths.parent
        assert database_grid_search.paths.fit.parent_id == parent_paths.identifier

    def test_is_grid_search(
            self,
            database_grid_search
    ):
        assert database_grid_search.paths.is_grid_search

from pathlib import Path

import pytest

import autofit as af
from autoconf.conf import output_path_for_test
from autofit.database.aggregator.scrape import Scraper

output_directory = Path(
    __file__
).parent / "output"


@pytest.fixture(
    name="parent_search"
)
@output_path_for_test(
    output_directory
)
def make_parent_search(model_gaussian_x1):
    search = af.m.MockSearch(
        name="parent"
    )
    search.paths.model = model_gaussian_x1
    return search


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
        search=af.m.MockOptimizer(
            session=session
        ),
        number_of_steps=2
    )
    search.fit(
        model=mapper,
        analysis=af.m.MockAnalysis(),
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


@output_path_for_test(
    output_directory
)
def test_scrape(
        grid_search,
        parent_search,
        model_gaussian_x1,
        session
):
    grid_search.fit(
        model=model_gaussian_x1,
        analysis=af.m.MockAnalysis(),
        parent=parent_search,
        grid_priors=[model_gaussian_x1.centre]
    )
    parent_search.fit(
        model=model_gaussian_x1,
        analysis=af.m.MockAnalysis()
    )
    parent_search.paths.save_all()

    Scraper(
        directory=output_directory,
        session=session
    ).scrape()

    aggregator = af.Aggregator(session)
    assert list(aggregator.query(
        aggregator.search.id == grid_search.paths.identifier
    ))[0].parent.id == parent_search.paths.identifier
    assert len(aggregator.values("max_log_likelihood")) > 0
    assert list(aggregator.grid_searches())[0].is_complete


@output_path_for_test(
    output_directory
)
def test_incomplete(
        grid_search,
        session
):
    grid_search.save_metadata()

    Scraper(
        directory=output_directory,
        session=session
    ).scrape()

    session.commit()

    aggregator = af.Aggregator(
        session
    )
    aggregator = aggregator(
        aggregator.search.is_complete
    )
    assert len(aggregator) == 0


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

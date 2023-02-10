import pytest

import autofit as af


@pytest.fixture(
    name="search"
)
def make_search(session):
    return af.SearchGridSearch(
        search=af.m.MockOptimizer(
            session=session
        ),
        number_of_steps=2
    )


@pytest.fixture(
    autouse=True
)
def run_search(
        search,
        mapper
):
    search.fit(
        model=mapper,
        analysis=af.m.MockAnalysis(),
        grid_priors=[
            mapper.component.one_tuple.one_tuple_0,
            mapper.component.one_tuple.one_tuple_1,
        ]
    )


def test_save_result(
        search
):
    assert isinstance(
        search.paths.load_object(
            "result"
        ),
        af.GridSearchResult
    )


def test_aggregate(
        session
):
    aggregator = af.Aggregator(
        session
    )

    grid_searches = aggregator.grid_searches()
    assert len(
        grid_searches
    ) == 1
    assert len(
        grid_searches.children()
    ) > 0


def test_aggregate_completed(
        session
):
    session.commit()
    aggregator = af.Aggregator(
        session
    )
    aggregator = aggregator(
        aggregator.search.is_complete
    )
    grid_searches = aggregator.grid_searches()
    assert len(
        grid_searches
    ) == 1

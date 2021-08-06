import pytest

import autofit as af
from autofit.mock.mock import MockAnalysis
from test_autofit.non_linear.grid.test_optimizer_grid_search import MockOptimizer


@pytest.fixture(
    name="previous_search"
)
def make_previous_search():
    return af.DynestyStatic()


@pytest.fixture(
    name="grid_search"
)
def make_grid_search(
        mapper,
        previous_search
):
    search = af.SearchGridSearch(
        search=MockOptimizer(),
        number_of_steps=2,
        previous_search=previous_search
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


def test_previous_search(
        grid_search,
        previous_search
):
    identifier = grid_search.paths.previous_search_identifier
    assert identifier == previous_search.paths.identifier


def test_is_grid_search(
        grid_search
):
    assert grid_search.paths.is_grid_search

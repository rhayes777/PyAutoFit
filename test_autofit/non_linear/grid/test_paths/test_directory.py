import autofit as af
from autofit.mock.mock import MockAnalysis
from test_autofit.non_linear.grid.test_optimizer_grid_search import MockOptimizer


def test_is_grid_search(
        mapper
):
    search = af.SearchGridSearch(
        search=MockOptimizer(),
        number_of_steps=2
    )
    search.fit(
        model=mapper,
        analysis=MockAnalysis(),
        grid_priors=[
            mapper.component.one_tuple.one_tuple_0,
            mapper.component.one_tuple.one_tuple_1,
        ]
    )
    assert search.paths.is_grid_search

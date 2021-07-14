import autofit as af
from autofit.mock.mock import MockAnalysis
from test_autofit.non_linear.grid.test_optimizer_grid_search import MockOptimizer


def test_save_result(
        mapper,
        session
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
        ]
    )
    assert isinstance(
        search.paths.load_object(
            "result"
        ),
        af.GridSearchResult
    )

import pytest

import autofit as af
from autofit import Result
from autofit.non_linear.grid.grid_search import ResultBuilder
from autofit.non_linear.grid.grid_search.job import JobResult
from autofit.non_linear.grid.grid_search.result_builder import Placeholder


@pytest.fixture(
    name="result_builder"
)
def make_result_builder():
    return ResultBuilder(
        lists=[[1], [2], [3]],
        grid_priors=[]
    )


@pytest.fixture(
    name="add_results"
)
def make_add_results(
        result_builder
):
    def add_results(*numbers):
        for number in numbers:
            result_builder.add(
                JobResult(
                    af.MockResult(
                        search=af.MockSearch(
                            name=str(number)
                        ),
                        model=af.Model(
                            af.Gaussian
                        )
                    ),
                    [],
                    number
                )
            )

    return add_results


def test_ordering(
        result_builder,
        add_results
):
    add_results(0, 1, 2)
    result = result_builder()
    assert [
               result.search.name
               for result in result.results
           ] == ['0', '1', '2']


def test_gaps(
        result_builder,
        add_results
):
    add_results(1)
    result = result_builder()

    first, second, third = result.results
    assert isinstance(first, Placeholder)
    assert isinstance(second, Result)
    assert isinstance(third, Placeholder)

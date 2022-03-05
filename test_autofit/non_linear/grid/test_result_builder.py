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
                    af.m.MockResult(
                        search=af.m.MockSearch(
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


@pytest.mark.parametrize(
    "numbers, t1, t2, t3",
    [
        (tuple(), Placeholder, Placeholder, Placeholder,),
        ((0,), Result, Placeholder, Placeholder,),
        ((1,), Placeholder, Result, Placeholder,),
        ((2,), Placeholder, Placeholder, Result,),
        ((1, 2,), Placeholder, Result, Result,),
    ]
)
def test_gaps(
        result_builder,
        add_results,
        numbers,
        t1, t2, t3
):
    add_results(*numbers)
    result = result_builder()

    first, second, third = result.results
    assert isinstance(first, t1)
    assert isinstance(second, t2)
    assert isinstance(third, t3)


def test_log_likelihoods(
        result_builder
):
    assert (result_builder().log_likelihoods_native == [
        None, None, None
    ]).all()


def test_best_result(
        result_builder,
        add_results
):
    add_results(1)
    assert isinstance(
        result_builder().best_result,
        Result
    )


def test_placeholder_samples():
    assert Placeholder().samples.anything is None

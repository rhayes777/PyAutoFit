import autofit as af
from autofit.non_linear.grid.grid_search import ResultBuilder
from autofit.non_linear.grid.grid_search.job import JobResult


def test_ordering():
    result_builder = ResultBuilder(
        lists=[[1], [2], [3]],
        grid_priors=[]
    )

    def add_job_result(
            number
    ):
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

    add_job_result(2)
    add_job_result(3)
    add_job_result(1)

    assert isinstance(
        result_builder(),
        af.GridSearchResult
    )
    result = result_builder()

    assert [
               result.search.name
               for result in result.results
           ] == ['1', '2', '3']
